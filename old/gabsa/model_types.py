seq2seq = ["t5","byt5","mt5","mbart"]
lm = ["xglm","gpt2"]

pd = {
    "t5" : {
        "Wikidepia/IndoT5-small" : "wid_small",
        "Wikidepia/IndoT5-base" : "wid_base",
        "Wikidepia/IndoT5-large" : "wid_large",
        "cahya/t5-base-indonesian-summarization-cased" : "cahya_summid",
        "panggi/t5-base-indonesian-summarization-cased" : "panggi_summid",
        "t5-small" : "en_small",
        "t5-base" : "en_base"
    },
    "byt5" : {
        "google/byt5-small" : "google_small",
        "google/byt5-base" : "google_base",
        "google/byt5-large" : "google_large"
    },
    "mt5" : {
        "google/mt5-small" : "google_small",
        "google/mt5-base" : "google_base",
        "google/mt5-large" : "google_large"
    },
    "mbart" : {
        "facebook/mbart-large-50" : "fb_large_50",
        "facebook/mbart-large-50-one-to-many-mmt" : "fb_mmt_large_50",

    },
    "xglm" : {
        "facebook/xglm-564M" : "fb_small",
        "facebook/xglm-1.7B" : "fb_base",
        "facebook/xglm-2.9B" : "fb_large",
        "facebook/xglm-4.5B" : "fb_xlarge"
    },
    "gpt2" : {
        "gpt2" : "gpt2"
    }
}