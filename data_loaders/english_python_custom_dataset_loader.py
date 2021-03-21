def get_english_python_custom_dataset(file_path):
    '''
	This function will load english python custom dataset
	:param: file_path: Path of file
	'''
    english_text_python_program_pair_list = []
    process_python_code=False
    i=1
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            #print(i)
            i += 1
            if process_python_code==False:
                if line.strip() == '':
                    continue
                if line.startswith('#'):
                    english_text = line
                    #english_text_list.append(line)
                    process_python_code=True
                    python_program=''
                else:
                    print(i, ": ", line)            
            else:
                if line.strip() == '':
                    process_python_code=False
                    english_text_python_program_pair_list.append((english_text, python_program))
                    python_program=''
                    english_text =''
                if line.lstrip().startswith('#') or line.lstrip().startswith('import'):
                    continue
                else:
                    python_program += line
					
    return english_text_python_program_pair_list