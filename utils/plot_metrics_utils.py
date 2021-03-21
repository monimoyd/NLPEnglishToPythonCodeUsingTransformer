import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_loss(train_loss_list, valid_loss_list):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_loss_list, label="Train Loss")
    axs.plot(valid_loss_list, label="Validation Loss")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
	
def plot_ppl(train_ppl_list, valid_ppl_list):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_ppl_list, label="Train Perplexity")
    axs.plot(valid_ppl_list, label="Validation Perplexity")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.show()
	
	
