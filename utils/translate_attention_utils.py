import math
import collections
import torch
from torchtext.data.utils import ngrams_iterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.clean_text_utils import clean_text

def translate_sentence(sentence, src_field, trg_field, trg_type_field, model, device, max_len = 250):
    model.eval()
    sent = clean_text(" ".join(sentence))
    sentence = sent.split()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    trg_type_indexes = [trg_type_field.vocab.stoi[trg_type_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        trg_type_tensor = torch.LongTensor(trg_type_indexes).unsqueeze(0).to(device)
        trg_type_mask = trg_mask
        
        with torch.no_grad():
            output, output_type, attention = model.decoder(trg_tensor, trg_type_tensor, enc_src, trg_mask, trg_type_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    trg_type_tokens = [trg_type_field.vocab.itos[i] for i in trg_type_indexes]
    
    return trg_tokens[1:], trg_type_tokens[1:], attention
	
def display_attention(sentence, translation, attention, n_heads = 6, n_rows = 3, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
	

def _compute_ngram_counter(tokens, max_n):
    """ Create a Counter with a count of unique n-grams in the tokens list
    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted
    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count
    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1,
             ('me', 'me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(' '))
                                         for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf
    Args:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)
    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus),\
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            if len(ngram) > 4:
                continue
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            if len(ngram) > 4:
                continue
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()
		
		
def calculate_bleu(data, src_field, trg_field, trg_type_field, model, device, max_len = 250):
    
    trgs = []
    pred_trgs = []
    
    for i in range(len(data.examples)):
        
        src = vars(data.examples[i])['English']
        trg = vars(data.examples[i])['Python']

        
        pred_trg, pred_trg_type, _ = translate_sentence(src, src_field, trg_field, trg_type_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        #trgs.append(trg)
        #print("trg=", trg)
        #print("pred_trg=", pred_trg)
        
    #return bleu_score(pred_trgs, trgs,  max_n=1, weights=[1.0] )
    return bleu_score(pred_trgs, trgs )
		
	
def translate_code(token_list):
    num_spaces = 0
    python_code_token_list = ['import pyforest\n']
    if "datetime" in token_list:
        python_code_token_list.append('import datetime\n')

    last_token = ''
    token_list_len = len(token_list)
    for i, token in enumerate(token_list):
        if token in ['<sos>', '<eos>']:
            continue
        elif token == 'NEWLINE':
            python_code_token_list.append('\n')
            if i+1 <token_list_len and token_list[i+1] not in ['INDENT','DEDENT']:
                space_token1 = ' ' * num_spaces
                python_code_token_list.append(space_token1)
        elif token == 'INDENT':
            num_spaces += 4
            if i+1 <token_list_len and token_list[i+1] not in ['INDENT']:
                space_token2 = ' ' * num_spaces
                python_code_token_list.append(space_token2)
        elif token == 'DEDENT':
            num_spaces -= 4
            if num_spaces < 0 :
                num_spaces = 0
            if num_spaces > 0 and  i+1 < token_list_len and token_list[i+1] not in ['DEDENT']:
                space_token3 = ' ' * num_spaces
                python_code_token_list.append(space_token3)
        else:
            if token in [".", '(', ')'] or last_token in ['NEWLINE', 'INDENT', 'DEDENT', '', '.', '(']:
                python_code_token_list.append(token)
            else:
                python_code_token_list.append(' ')
                python_code_token_list.append(token)

        last_token = token

    return ''.join(python_code_token_list)
	

def exec_code(python_code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try :
        exec(python_code)
    except:
        print("exec_code: Excpetion while executing code: ",python_code)
        traceback.print_exc()
        return ""
    finally:
        sys.stdout = old_stdout
    output = redirected_output.getvalue()
    return output	


	




