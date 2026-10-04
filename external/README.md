# external/  (= `$HN_EXTERNAL`)

Third-party datasets, code and released weights, downloaded by the user from their original sources.
Needed only for the raw-data route of the top-level README; `python reproduce/run_all.py` does not use them.

    ronin/source/                         RoNIN official code, commit 805b7f0f28bb164ce89ada9ac05a9470dbe3d715
    ronin_models/ronin_resnet/checkpoint_gsn_latest.pt
    ronin_models/ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt
    ronin_models/ronin_tcn/checkpoints/ronin_tcn_checkpoint.pt
    tlio_golden/                          TLIO data release (train_list.txt, val_list.txt, test_list.txt, sequence folders)
    IMUNet_dataset/                       IMUNet phone data (list_train.txt, list_test.txt, sequence folders)
    external_code/pylib_imunet/           (optional) site-packages providing numpy-quaternion, for the phone caches

Set the environment variable `HN_EXTERNAL` to use another location.
