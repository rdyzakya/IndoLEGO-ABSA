# DOE-Model Settings

## Dataset
* Restaurant 2015
* Restaurant 2016

## Model Candidates
* t5-base
* google/mt5-base
* google/flan-t5-base
* google/t5-v1_1-base
* google/t5-efficient-base
* google/long-t5-local-base
* facebook/bart-base
* openai-gpt
* google/byt5-base
* gpt2-medium
* Nicki/gpt3-base
* facebook/xglm-564M
* ai-forever/mGPT

## Pattern
```
{
    "template" : {
        "acos" : {
            "input" : "( <A> , <C> , <O> , <S> )",
            "output" : "( ASPECT , CATEGORY , OPINION , SENTIMENT )"
        },
        "aos" : {
            "input" : "( <A> , <O> , <S> )",
            "output" : "( ASPECT , OPINION , SENTIMENT )"
        },
        "acs" : {
            "input" : "( <A> , <C> , <S> )",
            "output" : "( ASPECT , CATEGORY , SENTIMENT )"
        },
        "ao" : {
            "input" : "( <A> , <O> )",
            "output" : "( ASPECT , OPINION )"
        },
        "as" : {
            "input" : "( <A> , <S> )",
            "output" : "( ASPECT , SENTIMENT )"
        },
        "cs" : {
            "input" : "( <C> , <S> )",
            "output" : "( CATEGORY , SENTIMENT )"
        }
    },
    "place_holder" : {
        "aspect" : "ASPECT",
        "opinion" : "OPINION",
        "category" : "CATEGORY",
        "sentiment" : "SENTIMENT"
    },
    "seperator" : ";",
    "categories" : ["CAT0","CAT1"],
    "mask" : {
            "(" : "/OB/",
            ")" : "/CB/",
            "," : "/AS/",
            ";" : "/ES/"
        }
}
```

## Prompt
```
"prompt_template" : {
    "extraction" : "Extract with the format PATTERN for the following text",
    "imputation" : "Impute the following IMPUTATION_FIELD for the following text"
}
```

## Early Stopping
Early stopping is not set (-1).