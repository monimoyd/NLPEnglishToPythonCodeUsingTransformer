import json
def get_conala_dataset(file_path):
    '''
	This function will load conala dataset
	:param: file_path: Path of file
	'''
	
    english_text_python_program_pair_list = []

    # Opening JSON file 
    with open(file_path) as f: 
        # returns JSON object as  
        # a dictionary 
        data = json.load(f) 
        for record in data:
            english_text_python_program_pair_list.append((record['intent'],record['snippet'] ))
            english_text_python_program_pair_list.append((record['rewritten_intent'],record['snippet'] ))
			
    return english_text_python_program_pair_list
	