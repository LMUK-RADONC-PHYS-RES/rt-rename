import ollama
import os
import openpyxl as xl
from openai import OpenAI
import csv
from time import sleep
from dash import set_props, html
import time
import base64
import io
import pandas as pd
import math
import pydicom
import fnmatch
import json

def load_guideline(nomenclature_xlsx,type='standard',description=False,regions=None):
    """
    Load guidelines from an Excel file based on the specified type and regions.
    
    Parameters:
    nomenclature_xlsx (str): Path to the Excel file containing the nomenclature guidelines.
    type (str): Type of guideline to load. Options are 'standard' or 'reverse'. Default is 'standard'.
    description (bool): Whether to include descriptions of the structures. Default is False.
    regions (list or None): List of regions to filter the guidelines. If None, all regions will be included. Default is None. 
    Allowed region strings: ['Thorax', 'Head and Neck', 'Abdomen', 'Limb', 'Pelvis', 'Body', 'Limbs'] 
    
    Returns:
    list: A list of dictionaries containing structure names and optionally descriptions.
    """
    #    
    workbook = xl.load_workbook(nomenclature_xlsx)
    
    if type == 'standard':
        column = 'F'
    elif type == 'reverse':
        column = 'G'
    else:
        print('Please provide a valid type: standard or reverse')
        return
    
    if regions == None:
        regions = []
        for row in workbook['TG263 v20170815']['D'][1:]:
            if row.value not in regions and row.value != None:
                regions.append(row.value)
    
    TG263_structures = []              
    for row in workbook['TG263 v20170815']['D']:
        if row.value in regions:
            struct_name = workbook['TG263 v20170815'][f'{column}{row.row}'].value
            if description == True:
                struct_desc = workbook['TG263 v20170815'][f'H{row.row}'].value
                TG263_structures.append({'name':struct_name,'description':struct_desc})
            else:
                TG263_structures.append({'name':struct_name})
    return TG263_structures


def load_structures_dir(dir_path,filter=None):
    """
    Load and filter structure files from a directory.
    Args:
        dir_path (str): The path to the directory containing structure files.
        filter (str, optional): A filter to apply to the structure files. 
                                If 'synthRAD2025', only files with the '.nrrd' 
                                extension are included, excluding those ending 
                                with '_stitched.nrrd', '_s2_def.nrrd', or '_s2.nrrd'.
    Returns:
        list: A list of structure file names without their extensions.
    """
    
    structs = os.listdir(dir_path)
    if filter=='synthRAD2025':
        structs = [struct for struct in structs if struct.endswith('.nrrd')]
        structs = [struct for struct in structs if not struct.endswith('_stitched.nrrd')]
        structs = [struct for struct in structs if not struct.endswith('_s2_def.nrrd')]
        structs = [struct for struct in structs if not struct.endswith('_s2.nrrd')]
    structures_center = [struct.split('.')[0] for struct in structs]
    # structures_center = ','.join(structures_center)
    # structures_center = structures_center.split(',')
    return structures_center

def get_prompts():
    """
    Returns all available prompt versions as a list.
    Returns:
        list: A list of available prompt versions.
    """
    prompts = os.listdir('./config')
    prompts = fnmatch.filter(prompts, 'prompt*.txt')
    return prompts
    

def parse_prompt(file_path, TG263_list, structure_input):
    """
    Parses a prompt template file and replaces placeholders with actual values.
    Args:
        file_path (str): The path to the prompt template file.
        TG263_list (list of dict): A list of dictionaries containing 'name' and optionally 'description' keys.
        structure_input (str): A string input that will have its whitespace replaced with underscores.
    Returns:
        str: The parsed prompt with placeholders replaced by actual values.
    """
    with open(file_path, 'r') as file:
        prompt_template = file.read()
    if 'name' and 'description' in TG263_list[0].keys():
        TG263_list = [f"        - Name: {item['name']}, Description: {item['description']}" for item in TG263_list]
    else:
        TG263_list = [f"        - Name: {item['name']}" for item in TG263_list]
    
    #remove whitespace from structure_input by splitting and joining with underscore
    structure_input = structure_input.split(' ')
    structure_input = '_'.join(structure_input)
    
    # Replace placeholders with actual values
    prompt = prompt_template.replace("{TG263_list}", '\n'.join(TG263_list))
    prompt = prompt.replace("{structure_input}", structure_input)
    
    return prompt

def parse_prompt_v2(file_path, TG263_list, structure_input):
    """
    Parses a prompt template file and replaces placeholders with provided values.
    Args:
        file_path (str): The path to the prompt template file.
        TG263_list (list): A list of dictionaries containing 'name' and 'description' keys.
        structure_input (str): A string input that will have its whitespace removed and replaced with underscores.
    Returns:
        str: The parsed prompt with placeholders replaced by actual values.
    """
    with open(file_path, 'r') as file:
        prompt_template = file.read()
        TG263_list = [f"        - Name: {item['name']}, Description: {item['description']}" for item in TG263_list]
    
    #remove whitespace from structure_input by splitting and joining with underscore
    structure_input = structure_input.split(' ')
    structure_input = '_'.join(structure_input)
    
    # Replace placeholders with actual values
    prompt = prompt_template.replace("{TG263_list}", '\n'.join(TG263_list))
    prompt = prompt.replace("{structure_input}", structure_input)
    
    return prompt

def sort_key(filename):
    return filename[0].lower()

def parse_filenames(filenames,tv_filter=True):
    """
    Parses a list of filenames, filters them based on specific criteria, and returns a list of dictionaries with metadata.

    Args:
        filenames (list of str): List of filenames to be parsed.
        tv_filter (bool, optional): If True, filters out filenames containing specific substrings related to TV structures. Defaults to True.

    Returns:
        list of dict: A list of dictionaries, each containing metadata about the parsed filenames. Each dictionary contains the following keys:
            - "local name" (str): The base name of the file without the extension.
            - "TG263 name" (str): An empty string placeholder for TG263 name.
            - "confidence" (str): An empty string placeholder for confidence.
            - "verify" (str): An empty string placeholder for verify.
            - "accept" (str): An empty string placeholder for accept.
            - "comment" (str): An empty string placeholder for comment.
    """
    nrrd_filenames = [
        struct.split(".")[0:-1]
        for struct in filenames
        if struct.endswith(".nrrd")
        and not any(
            suffix in struct
            for suffix in ["_stitched.nrrd", "_s2_def.nrrd", "_s2.nrrd"]
        )
    ]
    if tv_filter == "True":
        nrrd_filenames = [
            struct
            for struct in nrrd_filenames
            if not any(
                tv in struct[0] for tv in ["PTV", "GTV", "CTV", "ITV"])]
    nrrd_filenames = sorted(nrrd_filenames,key=sort_key)
    table_as_dicts = [
        {
            "local name": str(struct[0]),
            "TG263 name": str(),
            "confidence": str(),
            "verify": str(),
            "accept": str(),
            "comment": str(),
            "raw_output": str(),
            "timestamp": str(),
        }
        for struct in nrrd_filenames
    ]
    return table_as_dicts

def read_guideline(regions,guideline,description=True):
    """
    Reads and returns the nomenclature list based on the specified guideline.

    Args:
        regions (list): A list of regions to be included in the nomenclature.
        guideline (str): The guideline to be used. Options are 'TG263' or 'TG263_reverse'.
        description (bool, optional): Whether to include descriptions in the nomenclature. Defaults to True.

    Returns:
        list: A list of nomenclature items based on the specified guideline and regions.
    """
    #TODO: Add description selector to UI
    TG263_nomenclature = './config/TG263_nomenclature.xlsx'
    if guideline== 'TG263':
        nomenclature_list = load_guideline(TG263_nomenclature,'standard',description,regions)
    if guideline == 'TG263_reverse':
        nomenclature_list = load_guideline(TG263_nomenclature,'reverse',description,regions)
    return nomenclature_list

def parse_csv(contents,filename):
    """
    Parses a CSV file from its base64 encoded contents and extracts the first column into a list of dictionaries.
    Args:
        contents (str): The base64 encoded contents of the CSV file.
        filename (str): The name of the file being parsed.
    Returns:
        list: A list of dictionaries where each dictionary represents a row from the first column of the CSV file.
                Each dictionary contains the following keys:
                - "local name": The name of the structure with the '.nrrd' extension stripped.
                - "TG263 name": An empty string to be filled later.
                - "confidence": An empty string to be filled later.
                - "verify": An empty string to be filled later.
                - "accept": An empty string to be filled later.
                - "comment": An empty string to be filled later.
    """
    decoded = base64.b64decode(contents.split(',')[1])
    df = pd.read_csv(io.StringIO(decoded.decode()))
    structures = df.iloc[:,0].tolist()
    if len(df.columns) > 3:
        TG263_name = df.iloc[:,1].tolist()
        confidence = df.iloc[:,2].tolist()
        verify = df.iloc[:,3].tolist()
        accept = df.iloc[:,4].tolist()
        comment = df.iloc[:,5].tolist()
        raw_output = df.iloc[:,6].tolist()
    table_as_dicts = []
    for i, struct in enumerate(structures):
        if len(df.columns) > 3:
            acc = True if accept[i] == 'True' else False
            tg263_name = TG263_name[i]
            conf = confidence[i]
            verif = verify[i]
            comm = comment[i]
            raw = raw_output[i]
            table_as_dicts.append(
                {
                    "local name": struct.replace('.nrrd', ''),
                    "TG263 name": tg263_name,
                    "confidence": conf,
                    "verify": verif,
                    "accept": acc,
                    "comment": comm,
                    "raw output": raw,
                    "timestamp": str(),
                })
        else:
            table_as_dicts.append(
                {
                    "local name": struct.replace('.nrrd', ''),
                    "TG263 name": str(),
                    "confidence": str(),
                    "verify": str(),
                    "accept": str(),
                    "comment": str(),
                    "raw output": str(),
                    "timestamp": str(),
                })
    return table_as_dicts
        
def parse_dicom(contents, filename,tv_filter='False'):
    """
    Parses a DICOM file from its base64 encoded contents and extracts the structure names.
    
    Args:
        contents (str): The base64 encoded contents of the DICOM file.
        filename (str): The name of the DICOM file being parsed.
    
    Returns:
        list: A list of dictionaries where each dictionary represents a structure with the following keys:
            - "local name": The name of the structure with the '.dcm' extension stripped.
            - "TG263 name": An empty string to be filled later.
            - "confidence": An empty string to be filled later.
            - "verify": An empty string to be filled later.
            - "accept": An empty string to be filled later.
            - "comment": An empty string to be filled later.
    """
    decoded = base64.b64decode(contents.split(',')[1])
    dicom_file_path = f'/tmp/{filename}'
    with open(dicom_file_path, 'wb') as f:
        f.write(decoded)
    
    roi_names = read_dicom_rtstruct_names(dicom_file_path)
    
    if tv_filter == "True":
        roi_names = [struct for struct in roi_names if not any(tv in struct for tv in ["PTV", "GTV", "CTV", "ITV"])]

    roi_names = sorted(roi_names,key=sort_key)

    table_as_dicts = []
    for roi_name in roi_names:
        table_as_dicts.append(
            {
                "local name": roi_name,
                "TG263 name": str(),
                "confidence": str(),
                "verify": str(),
                "accept": str(),
                "comment": str(),
                "raw output": str(),
                "timestamp": str(),
            })
    
    return table_as_dicts

def read_dicom_rtstruct_names(dicom_file_path: str) -> list[str]:
    """
    Reads structure names from a DICOM RTSTRUCT file.

    Args:
        dicom_file_path (str): The path to the DICOM RTSTRUCT file.

    Returns:
        list[str]: A list of ROI names.
    """
    try:
        rtstruct = pydicom.dcmread(dicom_file_path)
        if "StructureSetROISequence" not in rtstruct:
            print(f"Error: StructureSetROISequence not found in {dicom_file_path}")
            return []
        
        roi_names = [roi.ROIName for roi in rtstruct.StructureSetROISequence]
        return roi_names
    except Exception as e:
        print(f"Error reading DICOM RTSTRUCT file {dicom_file_path}: {e}")
        return []

def write_dicom_rtstruct_names(dicom_file_path: str, new_names_map: dict[str, str], output_file_path: str | None = None):
    """
    Writes renamed structure names back to a DICOM RTSTRUCT file.

    Args:
        dicom_file_path (str): The path to the input DICOM RTSTRUCT file.
        new_names_map (dict[str, str]): A dictionary mapping original ROI names to new ROI names.
        output_file_path (str, optional): The path to save the modified DICOM RTSTRUCT file. 
                                         If None, the original file will be overwritten. Defaults to None.
    """
    try:
        rtstruct = pydicom.dcmread(dicom_file_path)
        if "StructureSetROISequence" not in rtstruct:
            print(f"Error: StructureSetROISequence not found in {dicom_file_path}")
            return

        modified = False
        for roi in rtstruct.StructureSetROISequence:
            if roi.ROIName in new_names_map:
                roi.ROIName = new_names_map[roi.ROIName]
                modified = True
        
        if modified:
            save_path = output_file_path if output_file_path else dicom_file_path
            rtstruct.save_as(save_path)
            print(f"Successfully updated ROI names in {save_path}")
        else:
            print(f"No matching ROI names found to update in {dicom_file_path}")

    except Exception as e:
        print(f"Error writing DICOM RTSTRUCT file {dicom_file_path}: {e}")

def get_models():
    """
    Retrieves the available models from the configuration file.
    
    Returns:
        list: A list of dictionaries containing model information.
              Each dictionary contains:
                - 'model_str': The string identifier for the model.
                - 'cloud': A string indicating if the model is cloud-based ('true' or 'false').
                - 'display_name': A user-friendly name for the model.
    """
    import json
    with open('./config/models.json', 'r') as f:
        models = json.load(f)

    model_names = [f'{model["name"]} | {model["parameters"]} | {"cloud" if model["cloud"] else "local"}' for model in models]
    return model_names

def get_model_str(model_name):
    """
    Retrieves the model string from the model name.
    
    Args:
        model_name (str): The name of the model in the format 'model_str | parameters | cloud/local'.
    Returns:
        str: The model string identifier.
    """
    print(f'Getting model string for {model_name}')
    with open('./config/models.json', 'r') as f:
        models = json.load(f)
    for model in models:
        if model_name == f'{model["name"]} | {model["parameters"]} | {"cloud" if model["cloud"] else "local"}':
            return model['model_str'], model["cloud"]
    return None

def run_model(model, prompt, guideline, region, structure_dict,column_defs=None,gui=True,uncertain=False):
    """
    Runs a specified model with given parameters and updates the structure dictionary with predictions.
    Args:
        model (str): The model to be used. Options can be configured in ./config/models.json
        prompt (str): The prompt version to be used. Supported versions include "v1", "v2", "v3", "v4", "v5", and "v6".
        guideline (str): The guideline to be used for reading nomenclature.
        region (str): The region for which the guideline is applicable.
        structure_dict (list): A list of dictionaries containing structure information.
        column_defs (list, optional): A list of column definitions for GUI. Defaults to None.
        gui (bool, optional): Flag to indicate if GUI updates are needed. Defaults to True.
    Returns:
        list: Updated structure dictionary with predictions, confidence scores, and verification status.
    """
    model_str, model_cloud = get_model_str(model)
    cloud = model_cloud
    print(f'Running model: {model_str} with cloud={cloud}')

    nomenclature_list = read_guideline(region,guideline,description=False)
    column_defs_updated = []
    if gui:
        for col in column_defs:
            if col['field'] == 'accept':
                col['cellRenderer'] = 'Checkbox'
                col['cellRendererParams'] = {'disabled': False}
                column_defs_updated.append(col)
            else:
                column_defs_updated.append(col)
            
    if gui:
        set_props('main-data-table', {'columnDefs': column_defs})
    for i,structure in enumerate(structure_dict):
        string = f"Model running {i+1}/{len(structure_dict)}..."
        #set_props('status-bar', {'children': html.P(string)})
        prompt_str = parse_prompt(f'./config/{prompt}',nomenclature_list,structure['local name'])
        system_prompt = '''You are a radiation oncology professional with vast experience in naming structures for radiotherapy treatment planning. You understand English, German and Dutch.
        You are tasked with renaming structures based on a standardized nomenclature list. This task is crucial for standardizing radiation oncology 
        practices across different institutions from different countries and improving data interoperability. Follow the prompts strictly and do not provide 
        any additional information.'''
        
        if uncertain:
            print('uncertainty')
            num_inference = 10
            structure_dict_inf = structure_dict[i].copy()
            structure_dict_list = []
            prediction_list = []
            for j in range(num_inference):
                if cloud:
                    response = run_llm_cloud(model=model_str,prompt=prompt_str,system_prompt=system_prompt,temperature=1,top_p=0.95)
                else:
                    response = run_llm(model=model_str,prompt=prompt_str,system_prompt=system_prompt,temperature=1,top_p=0.95)
                response_str = response['response'].split('</think>')[-1].splitlines()[-1]
                prediction = response_str.split(',')[0]
                if len(response_str.split(',')) > 1:
                    confidence = response_str.split(',')[1]
                else:
                    confidence = 'None'
                nomenclature_names = [name['name'] for name in nomenclature_list]
                structure_dict_inf["TG263 name"] = prediction
                structure_dict_inf["confidence"] = confidence
                structure_dict_inf["verify"] = check_TG263_name(nomenclature_names,prediction)
                structure_dict_inf["raw output"] = response['response']
                if structure_dict_inf["verify"] == "pass":
                    structure_dict_inf["accept"] = True
                else:
                    structure_dict_inf["accept"] = False
                structure_dict_list.append(structure_dict_inf)
                prediction_list.append(prediction)
            #find the most common prediction
            counts = {x:prediction_list.count(x) for x in set(prediction_list)}
            most_common_prediction = max(counts, key=counts.get)
            
            # get the most common prediction from the structure_dict_list
            for p in structure_dict_list:
                if p["TG263 name"] == most_common_prediction:
                    structure_dict[i] = p
                    break
        
            # calculate entropy
            counts_list = list(counts.values())
            entropy = -sum((count/num_inference) * math.log2(count/num_inference) for count in counts_list)
            print(f"Entropy: {entropy}")
            structure_dict[i]["entropy"] = f'{entropy:.3f}'
            structure_dict[i]["uncertainty_list"] = prediction_list
            structure_dict_inf["confidence"] = f'{entropy:.3f}'
                               
            
            if gui:
                set_props("main-data-table", {"rowData": structure_dict})
        else:
            if cloud:
                start = time.time()
                response = run_llm_cloud(model=model_str,prompt=prompt_str,system_prompt=system_prompt)
                end = time.time()
                #print(f"Inference time: {end-start}")
                
            else:
                start = time.time()
                response = run_llm(model=model_str,prompt=prompt_str,system_prompt=system_prompt,gui=gui)
                end = time.time()
                #print(f"Inference time: {end-start}")
            response_str = response['response'].split('</think>')[-1].splitlines()[-1]
            #print(response_str)
            prediction = response_str.split(',')[0]
            confidence = response_str.split(',')[1]
            nomenclature_names = [name['name'] for name in nomenclature_list]
            structure_dict[i]["TG263 name"] = prediction
            structure_dict[i]["confidence"] = confidence
            structure_dict[i]["verify"] = check_TG263_name(nomenclature_names,prediction)
            structure_dict[i]["raw output"] = response['response']
            if structure_dict[i]["verify"] == "pass":
                structure_dict[i]["accept"] = True
            else:
                structure_dict[i]["accept"] = False
            if gui:
                set_props("main-data-table", {"rowData": structure_dict})
            #except:
            #    print("Error in LLM")
    sleep(3)
    return structure_dict

def run_llm(model:str='llama3.1:70b-instruct-q4_0', prompt:str=None, system_prompt:str=None,temperature=0, top_p=0.1, gui=False):
    """
    Generates a response from a language model using the specified parameters.
    Args:
        model (str): The name of the model to use. Default is 'llama3.1:70b-instruct-q4_0'.
        prompt (str): The prompt to provide to the model. Default is None.
        system_prompt (str): The system prompt to provide to the model. Default is None.
    Returns:
        response: The generated response from the model.
    """
    ollama_client = ollama.Client(host = 'ollama:11434')
    try:
        response = ollama_client.generate(
                                model=model,
                               prompt=prompt,
                               system=system_prompt,
                               #format='json',
                               options={
                                    "seed": 111,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "num_ctx": 24000,
                               }
                               )
        return response
    except ollama._types.ResponseError as e:
        if gui:
            set_props("status-bar", {'children': html.P(f"Pulling {model} from ollama server...")})
        ollama_client.pull(model=model)
        response = ollama_client.generate(
                                model=model,
                               prompt=prompt,
                               system=system_prompt,
                               #format='json',
                               options={
                                    "seed": 111,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "num_ctx": 24000,
                               }
                               )
        return response
    except ollama.errors.OllamaError as e:
        print(f"Error pulling model {model}: {e}")
        return None
    

def run_llm_cloud(model:str='meta-llama/Meta-Llama-3.1-70B-Instruct-fast', prompt:str=None, system_prompt:str=None, temperature = 0, top_p = 0.9):
    """
    Sends a prompt to the specified language model hosted on Nebius AI and returns the generated response.
    Args:
        model (str): The identifier of the language model to use. Defaults to 'meta-llama/Meta-Llama-3.1-70B-Instruct-fast'.
        prompt (str): The user prompt to send to the language model.
        system_prompt (str): An optional system prompt to provide context or instructions to the language model.
    Returns:
        dict: A dictionary containing the response from the language model with the key 'response'.
    """
    try:
        client = OpenAI(
            base_url=os.environ.get("OPEN_AI_URL"),
            api_key=os.environ.get("OPEN_AI_API_KEY")
            )
    except Exception as e:
        print(f"For cloud inference you need to set OPEN_AI_API_KEY and OPEN_AI_API_URL environment vaiables: {e}")
        return None
    
    response = client.chat.completions.create(
        model=model,
        max_tokens=32000,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    

    message = {}
    message['response'] = response.choices[0].message.content
    return message

def create_output_csv(output_list,output_csv):
    """
    Writes a list of dictionaries to a CSV file.
    Args:
        output_list (list of dict): The list of dictionaries to write to the CSV file.
        output_csv (str): The file path where the CSV file will be created.
    Returns:
        None
    """
    keys = output_list[0].keys()
    with open(output_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(output_list)

def check_TG263_name(TG263_list,structure):
    """
    Check if a given structure name is present in the TG263 list.
    Args:
        TG263_list (list): A list of structure names following the TG263 standard.
        structure (str): The name of the structure to check.
    Returns:
        str: 'pass' if the structure is in the TG263 list, otherwise 'fail'.
    """

    if structure in TG263_list:
        return 'pass'
    else: 
        return 'fail'

def structure_dict_to_csv(structure_dict,output_csv):
    """
    Writes a list of dictionaries to a CSV file, with each dictionary representing a row.
    Args:
        structure_dict (list of dict): A list of dictionaries where each dictionary represents a row in the CSV file.
        output_csv (str): The file path for the output CSV file.
    Returns:
        None
    Notes:
        - The keys of the first dictionary in the list are used as the header of the CSV file.
        - The keys are capitalized to form the header row.
    """
    # write a list of dictionaries to a csv file, each list item is a row
    # capitalize the keys of the dictionary to make them the header
    keys = structure_dict[0].keys()
    with open(output_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(structure_dict)

def update_dicom(dicom_file, structure_dict):
    """
    Updates the DICOM RTSTRUCT file with new structure names based on the provided structure dictionary.
    
    Args:
        dicom_file (pydicom.Dataset): A pydicom file object representing the DICOM RTSTRUCT file to be updated.
        structure_dict (list of dict): A list of dictionaries containing structure information,
                                        where each dictionary has a 'local name' and 'TG263 name'.
    
    Returns:
        None
    """
    new_names_map = {struct['local name']: struct['TG263 name'] for struct in structure_dict if struct['accept']}
    if not new_names_map:
        print("No valid structure names to update in the DICOM file.")
        return
    # for each new_names_map go into dicom file and change the ROIName
    for roi in dicom_file.StructureSetROISequence:
        print(roi.ROIName)
        if roi.ROIName in new_names_map:
            print (f"Updating ROI name from {roi.ROIName} to {new_names_map[roi.ROIName]}")
            roi.ROIName = new_names_map[roi.ROIName]
    print(f"Updated DICOM RTSTRUCT file with new structure names.")
    return dicom_file

        