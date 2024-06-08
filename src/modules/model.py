import torch
from torch import nn
from torchvision import models
from .vocabulary import Vocabulary
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features: torch.Tensor = self.resnet(images)
        # [B, feature_maps, size1, size2] -> [B, size, feature_maps]
        features = features.flatten(start_dim=-2, end_dim=-1).movedim(-1, 1)
        return features


class BahdanauAttention(nn.Module):  
    def __init__(self, num_features, hidden_dim, output_dim=1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)
                
    def forward(self, features, decoder_hidden):
        # add additional dimension to a hidden (required for summation)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combine result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        atten_score = self.v_a(atten_tan)
        atten_weight = nn.functional.softmax(atten_score, dim = 1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        # the size of context equals to a number of feature maps
        context = torch.sum(atten_weight * features,  dim = 1)
        atten_weight = atten_weight.squeeze(dim=2)
        
        return context, atten_weight


class DecoderRNN(nn.Module):
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p=0.5):

        super(DecoderRNN, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5 
        
        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention) 
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # add attention layer 
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector 
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, captions, features):
        # create embeddings for captions of size (batch, seq_len, embed_dim)
        embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(embed.device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(embed.device)
        for t in range(seq_len):
            word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            word_embed = self.embeddings(output.argmax(-1)).squeeze(1) 
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, num_features, hidden_size, vocab_size):
        super().__init__()
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(
            num_features=num_features,
            embedding_dim=embed_size,
            hidden_dim=hidden_size,
            vocab_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs, atten_weights = self.decoderRNN(
            captions=captions,
            features=features
        )
        return outputs
    
    def caption_image(self, image, max_sentence=20):
        sentence = []
        weights = []
        features = self.encoderCNN(image.unsqueeze(0))
        input_word = torch.tensor(1).unsqueeze(0).cuda()
        h, c = self.decoderRNN.init_hidden(features)
        while True:
            embedded_word = self.decoderRNN.embeddings(input_word)
            context, atten_weight = self.decoderRNN.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.decoderRNN.lstm(input_concat, (h,c))
            h = self.decoderRNN.drop(h)
            output = self.decoderRNN.fc(h) 
            # scoring = torch.nn.functional.log_softmax(output, dim=1)
            # top_idx = scoring[0].topk(1)[1]
            sentence.append(output.argmax(-1).item())
            weights.append(atten_weight)
            input_word = output.argmax(-1)

            if (len(sentence) >= max_sentence or input_word.item() == 2):
                break
        return sentence