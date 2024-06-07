import torch
from torch import nn
from torchvision import models
from .vocabulary import Vocabulary
import numpy as np


class EncoderCNN(nn.Module):
    """Encoder inputs images and returns feature maps.
    Aruments:
    ---------
    - image - augmented image sample
    
    Returns:
    ---------
    - features - feature maps of size (batch, height*width, #feature maps)
    """
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        # [B, feature_maps, size1, size2] -> [B, size, feature_maps]
        features = torch.einsum('bfss -> bsf', features) 
        return features


class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
        Source: https://arxiv.org/pdf/1409.0473.pdf
     
    """    
    def __init__(self, num_features, hidden_dim, output_dim = 1):
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
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder
                
        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
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

    """Attributes:
        - embedding_dim - specified size of embeddings;
        - hidden_dim - the size of RNN layer (number of hidden states)
        - vocab_size - size of vocabulary 
        - p - dropout probability
    """
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

    def forward(self, captions, features, sample_prob = 0.0):
        
        """
        Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling
        
        Returns
        ----------
        - outputs - output logits from t steps
        - atten_weights - weights from attention network
        """
        # create embeddings for captions of size (batch, seq_len, embed_dim)
        embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(embed.device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(embed.device)
        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = nn.functional.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1) 
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder
    
        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0
    
    def greedy_search(self, features, max_sentence = 20):
        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)
            
        Returns:
        ----------
        - sentence - list of tokens
        """
        sentence=[]
        weights=[]

        input_word = torch.tensor(0).unsqueeze(0).cuda()
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h) 
            scoring = torch.nn.functional.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx

            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence, weights
    
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
    
    def caption_image(self, image, vocab: Vocabulary, max_len=20):
        result_caption = []
        with torch.no_grad():
            # encoded = self.encoderCNN(image.unsqueeze(0)).unsqueeze(1)
            # hidden = torch.zeros((self.decoderRNN.num_layers, 1, self.decoderRNN.hidden_size), device=encoded.device)
            # hidden = None
            features = self.encoderCNN(image.unsqueeze(0)).unsqueeze(1)
            # features = features.reshape(-1, 1, self.decoderRNN.embed_size)
            tgt = self.decoderRNN.word_embedding(torch.ones((1, 1), device=features.device, dtype=torch.int32))
            for i in range(max_len):
                # gru_out, hidden = self.decoderRNN.gru(embedding, hidden)
                # attn, _ = self.decoderRNN.attn(gru_out, gru_out, gru_out)
                tgt = self.decoderRNN.transformer(src=features, tgt=tgt)
                output = self.decoderRNN.linear(tgt)
                predicted = output.argmax(-1)
                result_caption.append(predicted.item())
                # embedding = self.decoderRNN.word_embedding(predicted)
                if vocab.idx2word[predicted.item()] == '<EOS>':
                    break
        # predicted = predicted.cpu().numpy().squeeze().tolist()
        # print(predicted)
        return [vocab.idx2word[idx] for idx in result_caption]
    
    def greedy_search(self, image, max_sentence=20):
        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)
            
        Returns:
        ----------
        - sentence - list of tokens
        """
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
            scoring = torch.nn.functional.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx

            if (len(sentence) >= max_sentence or top_idx == 2):
                break
        return sentence
    
    
# class Decoder(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super().__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
        
#         self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
#         # self.gru = nn.GRU(
#         #     self.embed_size, 
#         #     self.hidden_size, 
#         #     self.num_layers, 
#         #     batch_first=True
#         # )
        
#         self.linear = nn.Linear(self.hidden_size, self.vocab_size)
#         # self.dropout = nn.Dropout(0.3)
        
#         # self.attn = nn.MultiheadAttention(
#         #     embed_dim=self.embed_size,
#         #     num_heads=self.num_layers,
#         #     dropout=0.4,
#         #     batch_first=True
#         # )
#         self.transformer = nn.Transformer(
#             d_model=embed_size,
#             batch_first=True
#         )
    
#     def forward(self, features, captions):
#         embeddings = self.word_embedding(captions) # [B, seq_size, embed_size]
#         # features = features.reshape(-1, 1, embeddings.shape[-1]) # [B, seq_size, embed_size]
#         features = features.unsqueeze(1)
#         transformer_output = self.transformer(src=features, tgt=embeddings)
#         output = self.linear(transformer_output)
#         return output