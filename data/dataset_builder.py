import pandas as pd


def build_single_voice_dataset():
    data = pd.read_csv('./NISQA_results_deduplicated.csv')
    data['client_id'] = data['client_id'].apply(lambda x: x[:6])
    data = data.drop(columns=['sentence_id', 'up_votes', 'down_votes', 'model'])
    data.to_csv('./cv_17_single_full.tsv', sep='\t', index=False)


build_single_voice_dataset()
exit()


def build_dataset():
    data = pd.read_csv('./NISQA_results_deduplicated.csv', delimiter='\t')

    data['audio_length_minutes'] = data['audio_length'] / 60000
    data['client_id'] = data['client_id'].apply(lambda x: x[:6])
    data['quality'] = data[['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']].mean(axis=1)

    data_sorted = data.sort_values(by='mos_pred', ascending=False)

    grouped = data_sorted.groupby('client_id')

    filtered_clients = pd.DataFrame()
    # Iterate over each group
    for name, group in grouped:
        group['cumulative_audio_length_minutes'] = group['audio_length_minutes'].cumsum()
        valid_records = group[group['cumulative_audio_length_minutes'] <= 32]  # leave some margin for test data

        if valid_records['audio_length_minutes'].sum() >= 15:
            filtered_clients = pd.concat([filtered_clients, valid_records])

    filtered_clients['gender'] = filtered_clients['gender'].replace(
        {'male_masculine': 'male', 'female_feminine': 'female'})

    client_id_to_gender = {
        'd5bcc7': 'female',
        '2266fd': 'female',
        '30456c': 'female',
        'c5c18d': 'male',
        '06da81': 'female',
        'be3f03': 'female',
        '2a7d89': 'male',
        '332b0e': 'female',
        'eb5f38': 'female',
        'd63d1f': 'female',
        'd6944d': 'male',
        '94b2ac': 'female',
        '7b68c5': 'male',
        'f49187': 'female',
        '7b2e48': 'female',
        '30b015': 'female',
        '5658bb': 'female',
        'f7c950': 'female',
        'c39924': 'female',
        '28ff9b': 'female',
        'ffb8ae': 'female',
        '169982': 'female',
        'b4090f': 'male',
        '77fab6': 'female',
        '7806c2': 'female',
        '62d8ad': 'female',
        '7777f7': 'female',
        '126d55': 'male',
        '01afc6': 'female',
        '8f7663': 'female',
        '8443be': 'female',
        '7d160c': 'female',
        'cc89c4': 'female',
        '99e4c4': 'female',
    }

    filtered_clients.loc[filtered_clients['gender'].isnull(), 'gender'] = filtered_clients['client_id'].map(
        client_id_to_gender)

    # Calculate total audio length and average mos_pred for each client
    summary = filtered_clients.groupby('client_id').agg(
        length=('audio_length_minutes', 'sum'),
        avg_quality=('quality', 'mean'),
        avg_mos=('mos_pred', 'mean'),
        avg_noi=('noi_pred', 'mean'),
        avg_dis=('dis_pred', 'mean'),
        avg_col=('col_pred', 'mean'),
        gender=('gender', 'first'),
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # Print results for each client
        summary_by_length = summary.sort_values(by='length', ascending=False)
        print(summary_by_length)

        summary_by_mos = summary.sort_values(by='avg_mos', ascending=False)
        print(summary_by_mos)

        # Calculate and print the total audio length of the filtered section
        total_audio_length = filtered_clients['audio_length_minutes'].sum()
        total_mos_pred = filtered_clients['mos_pred'].mean()
        print("Total audio length of the filtered section:", total_audio_length / 60, "hours")
        print("Average MOS prediction of the filtered section:", total_mos_pred)

        # Female client count by gender column
        female_clients = filtered_clients[filtered_clients['gender'] == 'female']['client_id'].unique()
        male_clients = filtered_clients[filtered_clients['gender'] == 'male']['client_id'].unique()

        print(f"Females: {len(female_clients)}, Males: {len(male_clients)}")

        """
        When sorted by length:
        Total audio length of the filtered section: 57.80742166666666 hours
        Average MOS prediction of the filtered section: 3.4041115665535258

        When sorted by MOS:
        Total audio length of the filtered section: 57.810471388888885 hours
        Average MOS prediction of the filtered section: 3.6332488130555114
        """

    # Save the filtered dataset to a new TSV file
    filtered_clients = filtered_clients.drop(columns=['audio_length_minutes'])
    # filtered_clients['audio_length'] = filtered_clients['audio_length_real']
    filtered_clients.to_csv('./sets/common_voice/data_for_adobe.tsv', sep='\t', index=False)


def build_mini_dataset():
    data = pd.read_csv('./sets/common_voice/updated_validated_w_length_nisqa.tsv', delimiter='\t')
    data['client_id'] = data['client_id'].apply(lambda x: x[:6])

    # Get single client data
    data = data[data['client_id'] == '8b4d10']

    data['audio_length_minutes'] = data['audio_length'] / 60000
    data['quality'] = data[['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']].mean(axis=1)

    data = data.sort_values(by='mos_pred', ascending=False)

    data['cumulative_audio_length_minutes'] = data['audio_length_minutes'].cumsum()
    # data = data[data['cumulative_audio_length_minutes'] <= 120]

    data.to_csv('./8b4d10.tsv', sep='\t', index=False)


# build_dataset()
build_mini_dataset()
