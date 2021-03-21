import traceback
import tokenize
import io
import keyword
import spacy

spacy_en = spacy.load('en')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_python(text):
    """
    Tokenizes Python Code to list of strings
    """
    python_token_list = []
    raised_exception = False
    try:
        tokens = tokenize.tokenize(io.BytesIO(text.encode('utf-8')).readline)
        for five_tuple in tokens:
            if five_tuple.type == tokenize.COMMENT:
                continue
            elif five_tuple.type == tokenize.ENCODING:
                continue
            elif five_tuple.type == tokenize.INDENT:
                python_token_list.append("INDENT")
            elif five_tuple.type == tokenize.DEDENT:
                python_token_list.append("DEDENT")
            elif five_tuple.type == tokenize.NL or five_tuple.type == tokenize.NEWLINE:
                python_token_list.append("NEWLINE")
            elif five_tuple.type == tokenize.ENDMARKER :
                continue
            else:
                python_token_list.append(five_tuple.string)
    except Exception:
        raised_exception = True        
        print( "Exception: ", Exception, " program: ", text)
        traceback.print_exc()
    return python_token_list

def tokenize_python_type(text):
    """
    Tokenizes Python Code to list of strings representing type
    """
    python_type_list = []
    raised_exception = False
    last_token = ''
    i = 0
    try:
        tokens = tokenize.tokenize(io.BytesIO(text.encode('utf-8')).readline)
        for five_tuple in tokens:
            i += 1
            # print(i)
            if last_token == '(' and len(python_type_list) > 0 and  python_type_list[-1] == tokenize.NAME:
                python_type_list[-1] = 'FUNCTION'

            if five_tuple.type == tokenize.COMMENT:
                last_token = ''
                continue
            elif five_tuple.type == tokenize.ENCODING:
                last_token = ''
                continue
            elif five_tuple.type == tokenize.INDENT:                
                python_type_list.append("INDENT")
                last_token = ''
            elif five_tuple.type == tokenize.DEDENT:                
                python_type_list.append("DEDENT")
                last_token = ''
            elif five_tuple.type == tokenize.NL or five_tuple.type == tokenize.NEWLINE:
                python_type_list.append("NEWLINE")
                last_token = ''
            elif five_tuple.type == tokenize.ENDMARKER :
                last_token = ''
                continue
            elif five_tuple.type == tokenize.NAME:                
                if keyword.iskeyword(five_tuple.string):
                    python_type_list.append("KEYWORD")
                elif five_tuple.string.isidentifier():
                    python_type_list.append("IDENTIFIER")
                elif last_token == 'def':
                    python_type_list.append("FUNCTION_DECLARATION")
                else:
                    python_type_list.append('NAME')
                last_token = five_tuple.string
            else:
                python_type_list.append(tokenize.tok_name[five_tuple.type])
                last_token = five_tuple.string          

    except Exception:
        raised_exception = True
        print( "Exception: ", Exception, " program: ", text)
        traceback.print_exc()
    return python_type_list