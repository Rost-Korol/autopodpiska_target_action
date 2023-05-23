import pandas as pd
import pickle

def join_data(df_x, df_y):
    target_action_list = [
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ]

    df_y['target'] = df_y['event_action'].apply(lambda x: 1 if x in target_action_list else 0)
    pivot_hits = pd.pivot_table(
        df_y,
        index='session_id',
        values='target',
        aggfunc='sum'
    )
    pivot_hits['target'] = pivot_hits['target'].apply(lambda x: 1 if x > 1 else x)

    return df_x.merge(pivot_hits, how='inner', left_on='session_id', right_index=True)

with open('ga_sessions.pkl', 'rb') as file:
    df_x = pickle.load(file)

with open('ga_hits-001.pkl', 'rb') as file:
    df_y = pickle.load(file)

with open('join_data.pkl', 'wb') as file:
    pickle.dump(join_data(df_x, df_y), file)







