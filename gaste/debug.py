import preprocessing
import data_utils
import torch
from transformers import XGLMTokenizer
import pandas as pd

def debug1():

    line = "Mereka juga berterima kasih kepada Kim karena telah memberikan ' kebahagiaan ' .####[([5], [2, 3], 'POS'), ([5], [7, 8, 9, 10, 11], 'POS')]"
    splitted_line = line.split("####")
    text = splitted_line[0]
    num_target = eval(splitted_line[1])

    target = preprocessing.process_numtargets(text,num_target,"aste")
    stringified_target = preprocessing.stringify_target(text, num_target, target, "extraction", preprocessing.Pattern(task="aste"))
    stringified_target = stringified_target.replace(' ','')
    inverse_stringified_target = preprocessing.inverse_stringify_target(stringified_target,"extraction",preprocessing.Pattern(task="aste"))
    print("Text :",text)
    print("Num target :",num_target)
    print("Target :",target)
    print("Stringified target :",stringified_target)
    print("Inverse stringified target :",inverse_stringified_target)

def debug2():
    base_path = "/srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format/"
    paths = [base_path + "train_news_annotator.txt"]
    data = data_utils.read_files(paths,"aste")
    print(data)
    data.to_csv("ContohHasil.csv")

def debug3():
    base_path = "/srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format/"
    train_paths = [base_path + "train_news_annotator.txt", base_path + "train_news_student.txt", base_path + "train_socmed_twenty_percent.txt"]
    dev_paths = [base_path + "test_news.txt"]
    test_paths = [base_path + "test_news.txt"]

    dataset = data_utils.build_gabsa_dataset(train_paths,dev_paths,test_paths,"aste",0.0,42)
    print(dataset)

def debug4():
    line = "Mereka juga berterima kasih kepada Kim karena telah memberikan ' kebahagiaan ' .####[([5], [2, 3], 'POS'), ([5], [7, 8, 9, 10, 11], 'POS')]"
    splitted_line = line.split("####")
    text = splitted_line[0]
    num_target = eval(splitted_line[1])

    target = preprocessing.process_numtargets(text,num_target,"aste")
    stringified_target = preprocessing.stringify_target(text, num_target, target, "extraction", preprocessing.Pattern(task="aste"))
    prompt = "ekstraksi triplet aste :"

    prompted_text = text + " " + prompt

    text_input = prompted_text + " " + stringified_target
    tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-564M',padding_side="left")
    encoded_input = tokenizer.encode_plus(text_input,max_length=256,pad_to_max_length=True)
    decoded_input = tokenizer.decode(encoded_input["input_ids"],skip_special_tokens=True)

    encoded_prompted_text = tokenizer.encode_plus(prompted_text,max_length=256,pad_to_max_length=True)
    decoded_prompted_text = tokenizer.decode(encoded_prompted_text["input_ids"],skip_special_tokens=True)

    result = decoded_input[len(decoded_prompted_text):].strip()
    inverse_stringified_target = preprocessing.inverse_stringify_target(result,"extraction",preprocessing.Pattern(task="aste"))
    print("Prompted text (original) :",prompted_text)
    print("Text input (original):",text_input)
    print("Decoded prompted text :",decoded_prompted_text)
    print("Decoded text input :",decoded_input)
    print("Result :",result)
    print("Inverse stringified result :",inverse_stringified_target)

def debug5():
    base_path = "/srv/nas_data1/text/randy/aste/facebook-absa/data/interim/gaste_format/"
    train_paths = [base_path + "train_news_annotator.txt", base_path + "train_news_student.txt", base_path + "train_socmed_twenty_percent.txt"]
    dev_paths = [base_path + "test_news.txt"]
    test_paths = [base_path + "test_news.txt"]

    dataset = data_utils.build_gabsa_dataset(train_paths,dev_paths,test_paths,"aste",0.0,42)

    # dataset["train"]["prompt"] = "ekstraksi triplet aste :"
    # dataset["dev"]["prompt"] = "ekstraksi triplet aste :"
    # dataset["test"]["prompt"] = "ekstraksi triplet aste :"
    # print(dataset)
    prompter = preprocessing.Prompter("/srv/nas_data1/text/randy/aste/facebook-absa/gaste/prompts/aste.txt")
    result_text, result_prompts = prompter.add_prompt(dataset["dev"]["text"],prompt_side="right",option="random")
    print("Result text :",result_text[:5])
    print("Result prompts :",result_prompts[:5])
if __name__ == "__main__":
    # debug1()
    # debug2()
    # debug3()
    # debug4()
    debug5()