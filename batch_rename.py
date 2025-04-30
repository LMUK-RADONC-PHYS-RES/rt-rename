import utils
import base64
from tqdm import tqdm

input_files = ['/workspace/code/rt-rename/data/mc_uncertainty/correct_L3R.csv']
output_files = ['/workspace/code/rt-rename/data/mc_uncertainty/correct_L3R_uncertain.csv']

for k, input_file in enumerate(input_files):

    #read file and base64 encode (required for parser)
    file = open(input_files[k], 'rb')
    file_read = file.read()
    file_64_encode = base64.b64encode(file_read).decode('utf-8')

    # parse csv file into structure_dict format
    structure_dict = utils.parse_csv(f',{file_64_encode}',input_files[k])

    print(structure_dict)

    # split structure_dict into batches, run inference on each batch, and update structure_dict
    batchsize = 1
    batches = [structure_dict[i:i + batchsize] for i in range(0, len(structure_dict), batchsize)]
    for i, batch in tqdm(enumerate(batches),total=len(batches),desc='Processing batches'):
        processed_batch = utils.run_model(
            model='L3R-cloud',
            prompt='v3',
            guideline='TG263',
            region=None,
            structure_dict=batch,
            gui=False,
            uncertain=True
        )
        for j, processed_dict in enumerate(processed_batch):
            structure_dict[i * batchsize + j] = processed_dict
        utils.structure_dict_to_csv(structure_dict, output_files[k])
    