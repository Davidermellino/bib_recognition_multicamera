import numpy as np
import pandas as pd
import cv2
import Levenshtein


def select_bib_number(track_df):
    """
       Find the best number and the confidence of bib number detection of a runner.

       Args:
           track_df: detection of a single runner (identificate by trackid)

       Returns:
           number: the best bib number
           confidence: the confidence of the number prediction
    """

    numbers_as_int = track_df['number'].fillna(0).astype(int)

    filtered = track_df[
        (track_df['confidence'] >= 0.7) &
        (numbers_as_int >= 1000) &
        (numbers_as_int <= 9999)
    ]

    if len(filtered) > 0:
        max_conf_idx = filtered['confidence'].idxmax()
    else:
        max_conf_idx = track_df['confidence'].idxmax()

    number = track_df.loc[max_conf_idx]['number']
    confidence = track_df.loc[max_conf_idx]['confidence']

    return number, confidence

def aggregate_features(track_df, with_frame = True, final=False):
    """
       Aggregate the categorical and bib number features of all runner(trackid) detection

       Args:
           track_df: detection of a single runner (identificate by trackid)

       Returns:
           pd.Series(agg_dict): pandas Series of the aggregated feature
    """

    agg_dict = {}
    if not with_frame: #used for final aggregation ( when i have match_id and final_match_id)
        if final:
            agg_dict['final_match_id'] = track_df['final_match_id'].iloc[0]
            agg_dict['FD_number'], agg_dict['TI_number'], agg_dict['TD_number'], agg_dict['TI_number'] = None, None, None, None

            group_cams = track_df.groupby('cam')
            for cam, group in group_cams:
                if cam == 'back':
                    agg_dict['TD_number'] = group['TD_number'].iloc[0]
                    agg_dict['TI_number'] = group['TI_number'].iloc[0]
                else:
                    agg_dict['FD_number'] = group['FD_number'].iloc[0]
                    agg_dict['FI_number'] = group['FI_number'].iloc[0]
        else:
            agg_dict['match_id'] = track_df['match_id'].iloc[0]
    else:
        agg_dict['frames'] = track_df['nframe'].to_list()
        agg_dict['start_frame'] = track_df['nframe'].min()
        agg_dict['end_frame'] = track_df['nframe'].max()
        agg_dict['trackid'] = track_df['trackid'].iloc[0]
        agg_dict['cam'] = track_df['cam'].iloc[0]

    categorical_cols = [
        'gender', 'beard', 'hair_length', 'hair_color', 'glasses',
        'sunglasses', 'eyes_color', 'shirt_color', 'sleeves',
        'pulleover_etc', 'pants_color', 'bib_color'
    ]

    for col in categorical_cols:
        if col in track_df.columns:
            most_common = track_df[col].mode()
            if not most_common.empty:
                agg_dict[col] = most_common.iloc[0]
            else:  # Caso di colonna vuota
                agg_dict[col] = None

    if not final:
        group_cams = track_df.groupby('cam')
        for cam, group in group_cams:
            if not group['number'].isna().all():
                agg_dict[f'{group['cam'].iloc[0]}_number'], _ = select_bib_number(group)
            else:
                agg_dict[f'{group['cam'].iloc[0]}_number'] = None

    if not track_df['number'].isna().all():
        number, confidence = select_bib_number(track_df)
    else:
        number, confidence = None, None

    agg_dict['number'] = number
    agg_dict['confidence'] = confidence

    return pd.Series(agg_dict)


def temporal_overlap(track_a, track_b, offset, threshold, iou_thresh=0.1 ):
    """
       Checks if 2 trace are temporally near

       Args:
           track_a: first track
           track_b: second track
           threshold: threshold for overlap tollerance

       Returns:
           overlap: True or False
    """
    start_a, end_a = track_a['start_frame'], track_a['end_frame']
    start_b, end_b = track_b['start_frame'], track_b['end_frame']

    start_a -= offset
    end_a -= offset

    start_a_expanded = start_a - threshold
    end_a_expanded = end_a + threshold
    start_b_expanded = start_b - threshold
    end_b_expanded = end_b + threshold

    I = max(0, min(end_a_expanded, end_b_expanded) - max(start_a_expanded, start_b_expanded))
    U = max(end_a_expanded, end_b_expanded) - min(start_a_expanded, start_b_expanded)

    iou = I / U if U > 0 else 0.0
    return iou >= iou_thresh


def compute_similarity(tracka, trackb, dynamic_metric):
    """
      Compute the similarity of two tracks (previously aggregated with aggregate_features funtion)
      for each feature, 1 is given if the values are the same, 0 otherwise. The normalization is applied
      divinding by the number of the features

      For the number similarity, check if the numbers exists, then count the common digits and normalize dividing
      by the longest digit

      The final similarity score is computed by a weighted sum of the two similarity scores (categorical, number)

      Args:
          track_a: first track
          track_b: second track

      Returns:
          similarity: final similarity (between 0-1)
    """

    #CATEGORIAL SIMILARITY
    cat_features = ['gender', 'shirt_color', 'pants_color', 'glasses', 'hair_color','hair_length','sleeves','pulleover_etc','beard']
    matches = sum(tracka[f] == trackb[f] for f in cat_features) #number of matches

    cat_score = matches / len(cat_features) #normalize by number of features

    #NUMBER SIMILARITY
    #convert into string
    num1 = str(int(tracka['number'])) if pd.notna(tracka['number']) else ''
    num2 = str(int(trackb['number'])) if pd.notna(trackb['number']) else ''
    distance = Levenshtein.distance(num1, num2)
    max_len = max(len(num1), len(num2)) or 1
    number_score = 1 - (distance / max_len)

    #FINAL SIMILARITY
    if dynamic_metric:
        conf_a = tracka['confidence']
        conf_b = trackb['confidence']
        conf_a = 0.0 if conf_a is None or pd.isna(conf_a) else conf_a
        conf_b = 0.0 if conf_b is None or pd.isna(conf_b) else conf_b
        min_conf = min(conf_a,conf_b)
        similarity = (1-min_conf)*cat_score + min_conf*number_score

    else:
        similarity = 0.7*cat_score + 0.3*number_score

    return similarity


def build_similarity_matrix(tracks_cam1, tracks_cam2, offset=0, threshold=30, dynamic_metric=True):
    """

         Create similarity matrix of
            row -> number len of the first tracked cam (all runners aggregate features)
            cols -> number len of the second tracked cam (all runners aggregate features)

         Args:
             tracks_cam1: first cam tracks (all the aggregate tracks)
             tracks_cam2: second cam tracks (all the aggregate tracks)

         Returns:
             1 - similarity_matrix: similarity matrix with complement values (needed for hungarian algorithm)
       """

    n1 = len(tracks_cam1)
    n2 = len(tracks_cam2)
    similarity_matrix = np.zeros((n1, n2)) # initialize matrix

    # for i in range(len(tracks_cam1)):
    #     similarity_matrix[i+1,0] = -(tracks_cam1[i]['trackid'] - 1)
    #
    # for i in range(len(tracks_cam2)):
    #     similarity_matrix[0,i+1] = -(tracks_cam2[i]['trackid'] - 1)

    for i in range(n1):
        for j in range(n2):
            if temporal_overlap(tracks_cam1[i], tracks_cam2[j], iou_thresh=0.1, offset=offset, threshold=threshold):
                similarity_matrix[i, j ] = compute_similarity(tracks_cam1[i], tracks_cam2[j], dynamic_metric=dynamic_metric)

    return 1 - similarity_matrix #complement values (needed for hungarian algorithm)


def annotate_frame(frame, row, get_color_for_id):
    """
    Draws bounding boxes and labels on a frame for a single runner.

    Args:
        frame (np.ndarray): The video frame to annotate.
        row (pd.Series): A row containing the runner's annotation data
                         (columns: x1r, y1r, x2r, y2r, x1b, y1b, x2b, y2b, match_id, number).
        get_color_for_id (callable): Function that returns a color for a given match_id.

    Returns:
        np.ndarray: Annotated frame.
    """
    # --- Extract coordinates ---
    x1r, y1r, x2r, y2r = int(row["x1r"]), int(row["y1r"]), int(row["x2r"]), int(row["y2r"])
    if pd.notna(row["x1b"]):
        x1b, y1b, x2b, y2b = int(row["x1b"]), int(row["y1b"]), int(row["x2b"]), int(row["y2b"])

    # --- Extract info ---
    number = str(row.get("number", ""))
    match_id = row.get("match_id", "")
    color = get_color_for_id(match_id)

    # === Draw runner box ===
    cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), color, 2)

    # === Draw bib box ===
    if pd.notna(row["x1b"]):
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)

    # === Draw bib number above the bib box ===
    if pd.notna(row["x1b"]):
        text = f"{number}"
        text_x, text_y = x1b, max(20, y1b - 10)
        # Draw colored background for text readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        cv2.rectangle(frame, (text_x, text_y - th - 4), (text_x + tw + 4, text_y + 4), color, -1)
        # White text over colored background
        cv2.putText(frame, text, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # === Draw match ID above runner box ===
    id_text = f"mid:{match_id}"
    cv2.putText(frame, id_text, (x1r, max(20, y1r - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    return frame
