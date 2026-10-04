# data/

Evaluation and training caches rebuilt from the public datasets (RoNIN, RIDI, TLIO, IMUNet phone data).
They are **not needed** for `python reproduce/run_all.py`, which works from the archived results alone.
They are needed only to re-run inference or training ("Starting from the raw datasets" in the top-level README).

Expected layout after running the cache builders:

    data/train/ronin/{train,val}__<seq>.npz        88 files (SHA-256 in provenance/MANIFEST.sha256)
    data/eval/benchmark/{ronin,ridi}__<seq>.npz    RoNIN-seen and RoNIN-unseen (32 + 32) and RIDI (94)
    data/eval/tlio_posthoc/*.npz                   TLIO-test (36)
    data/eval/imunet_owndata/imunet__<seq>.npz     Phone-test (36)
    data/eval/confirm_tlio/, data/eval/confirm_imunet/   TLIO-confirm (318) and Phone-confirm (87)
