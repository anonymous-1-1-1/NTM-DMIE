import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
#___________Definition of hyperparameters_____
embed_size = 32
batch_size = 128
num_layers = 2
hidden_size = 32
output_size = 48
latent_dim = 10
categorical_dim = 10
temp=0.5
learning_rate = 1e-3
word_length = vocab_length + 1
#__________gpu & dataset loading__________
embedding_matrix = np.load("embedding_matrix.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = TextDataSet("InfomaxData.txt")
trainloader =  torch.utils.data.DataLoader(trainset,batch_size=128,shuffle=False,drop_last=True,num_workers=8)
#__________Neural Network Body___________
class InfomaxTextVAE(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,num_layers,latent_dim,categorical_dim,word_length,bidirectional=True):
        super(InfomaxTextVAE,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding_matrix = embedding_matrix
        self.bidirectional=bidirectional
        self.num_layers=num_layers
        self.output_size = output_size
        self.vocab_length = vocab_length
        self.embedding= nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))
        self.lstm=nn.LSTM(embed_size, hidden_size, num_layers,bidirectional=bidirectional)
        self.encode_fc = nn.Linear(hidden_size * 4,output_size)
        self.encode_fc2 = nn.Linear(output_size,latent_dim * categorical_dim)
        self.global_fc1 = nn.Linear(latent_dim * categorical_dim,10)
        self.global_fc2 = nn.Linear(10,1)
        self.local_fc1 = nn.Linear(latent_dim * categorical_dim,10)
        self.local_fc2 = nn.Linear(10,1)
        self.decode_fc = nn.Linear(latent_dim * categorical_dim,word_length)
        
        
    
    def encode(self,x):
        embeds = self.embedding(x)
        embeds = embeds.to(torch.float32)
        embeds=embeds.permute(1, 0, 2)
        h0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size).to(device)#2 for bidirectional
        c0 = torch.zeros(self.num_layers*2,batch_size,self.hidden_size).to(device)
        lstm_out,(final_hidden_state,final_cell_state) = self.lstm(embeds, (h0,c0))#final_hidden_state.size() = (1, batch_size, hidden_size)
        final_out = torch.cat((lstm_out[0], lstm_out[-1]), -1)
        out = self.encode_fc(final_out)
        out = self.encode_fc2(out)
        out = self.sigmoid(out)
        return embeds,out
        
    def gumbelTransfer(self,out):
        out = self.encode(x)
        out_dimTrans = out.view(out.size(0),latent_dim,categorical_dim)
        z = gumbel_softmax(out_dimTrans,temp)
        return z
        
    def globalDiscriminator(self,z):
        z = self.global_fc1(z)
        global_score = self.global_fc2(z)
        return global_score
        
    def localDiscriminator(self,z,w2v):
        local_feature = torch.cat(w2v,z)
        out = self.local_fc1(local_feature)
        local_score = self.local_fc2(out)
        return local_score
        
    def decode(self,z):
        renconst_z = self.decode_fc(z)
        return renconst_z
        
        
    def forward(self,x):
        embeds,out = self.encode(x)
        z = self.gumbelTransfer(out)
        global_score = self.globalDiscriminator(z)
        local_score = self.localDiscriminator(z)
        renconst_z = self.decode(z)

model=InfomaxTextVAE(embed_size,hidden_size,output_size,num_layers,latent_dim,categorical_dim,word_length,bidirectional=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1, last_epoch=-1)#changing the learning_rate through training
