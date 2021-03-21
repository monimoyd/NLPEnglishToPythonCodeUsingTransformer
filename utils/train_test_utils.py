import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.English
        trg = batch.Python
        trg_type = batch.PythonType
        
        optimizer.zero_grad()
        
        output, output_type, _ = model(src, trg[:,:-1],  trg_type[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss1 = criterion(output, trg)

        output_type_dim = output_type.shape[-1]
            
        output_type = output_type.contiguous().view(-1, output_type_dim)
        trg_type = trg_type[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss2 = criterion(output_type, trg_type)

        loss =  1.5 * loss1 + loss2
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
	
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.English
            trg = batch.Python
            trg_type = batch.PythonType

            output, output_type, _ = model(src, trg[:,:-1], trg_type[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss1 = criterion(output, trg)

            output_type_dim = output_type.shape[-1]
            
            output_type = output_type.contiguous().view(-1, output_type_dim)
            trg_type = trg_type[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss2 = criterion(output_type, trg_type)
            
            loss = 1.5 * loss1 +  loss2
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)