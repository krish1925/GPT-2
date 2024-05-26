# Building and training a bigram language model
from functools import partial
import math

import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange


class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.
    """

    def __init__(self, config):
        """ 
        Initialize the bigram language model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        super().__init__()
        # ========= TODO : START ========= #


        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)


        # ========= TODO : END ========= #
        
        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 1) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """

        # ========= TODO : START ========= #

        embeddingss = self.embeddings(x)
        embeddingss = self.dropout(embeddingss)
        logits = self.linear(embeddingss)

        return logits

        # ========= TODO : END ========= #

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling
        but instead of taking the token with the highest probability, we sample the next token from a multinomial distribution.


        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """

        ### ========= TODO : START ========= ###
        device = next(self.parameters()).device  
        context = context.unsqueeze(0).to(device)  
        
        for _ in range(max_new_tokens):
            logits = self.forward(context)
            logits = logits[:, -1, :] 
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()  
            
            next_token_tensor = torch.tensor([[next_token]], device=device) 
            context = torch.cat((context, next_token_tensor), dim=1) 

        return context.squeeze(0)
    

        ### ========= TODO : END ========= ###

class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=512,
    ):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.

         NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        causal_mask = None  # You have to implement this, currently just a placeholder

        # ========= TODO : START ========= #

        # self.key = ...
        # self.query = ...
        # self.value = ...
        # self.dropout = ...
        self.key = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.query = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.value = nn.Linear(input_dim, self.output_value_dim, bias=False)
        self.dropout = nn.Dropout()
        mask_tensor =torch.full((max_len, max_len),float('-inf'))
        causal_mask = torch.triu(mask_tensor)
        # ========= TODO : END ========= #

        self.register_buffer(
            "causal_mask", causal_mask
        )  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #
        # get the query, key and value vectors
        _, num_tokens, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        causal_mask = self.causal_mask[:num_tokens,:num_tokens]
        scores = torch.bmm(q, k.transpose(1, 2)) /(self.output_key_query_dim**0.5)
        masked_scores = scores.masked_fill(causal_mask==1, float('-inf'))
        attn = nn.functional.softmax(masked_scores, dim=-1)

        attn = self.dropout(attn)
        out = torch.bmm(attn, v)

        return out

        # ========= TODO : END ========= #
class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        
        # ========= TODO : START ========= #

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_key_query_dim = input_dim // num_heads
        self.output_value_dim = input_dim // num_heads

        for i in range(num_heads):
            setattr(self, f'head_{i}', SingleHeadAttention(input_dim, self.output_key_query_dim, self.output_value_dim, dropout))
        self.out = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        num_heads = self.num_heads
        head_concat = []
        for i in range(num_heads):
            single_head_attention = getattr(self, f'head_{i}')
            head_individual = single_head_attention(x)
            head_concat.append(head_individual)
        
        multi_head_out = torch.cat(head_concat, dim=-1)
        out = self.out(multi_head_out)
        out = self.dropout(out)
        return out\

        # ========= TODO : END ========= #


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        # ========= TODO : START ========= #

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        ### ========= TODO : START ========= ###

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x

        ### ========= TODO : END ========= ###


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        output = nn.functional.layer_norm(input, self.normalized_shape, self.gamma, self.beta, self.eps)
        return output
    

        # ========= TODO : END ========= #


class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        # ========= TODO : START ========= #

        # self.norm1 = ...
        # self.attention = ...
        # self.norm2 = ...
        # self.feedforward = ...

        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim)


        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        x1 = self.norm1(x)
        attention_output = self.attention(x1)
        x2 = x + attention_output 

        x3 = self.norm2(x2)
        feedforward_output = self.feedforward(x3)
        x4 = x2 + feedforward_output  

        return x4
    

        # ========= TODO : END ========= #


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings. 
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(
            config.embed_dim, config.vocab_size
        )  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        """

        ### ========= TODO : START ========= ###

        batch_size, seq_len = x.shape
        pos = self.pos[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
        x = self.vocab_embedding(x) + self.positional_embedding(pos)
        x = self.embed_dropout(x)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        x = self.prehead_norm(x)
        x = self.head(x)
        return x

        ### ========= TODO : END ========= ###

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Please copy the generate function from the BigramLanguageModel class you had implemented earlier.
        """

        ### ========= TODO : START ========= ###

        generated_sequence = context

        for _ in range(max_new_tokens):
            generated_sequence = generated_sequence.reshape((1, -1))
            if generated_sequence.shape[1] >= self.pos.shape[0]:
                truncated_context = generated_sequence[:, -self.pos.shape[0]:]
            else:
                truncated_context = generated_sequence

            output_logits = self.forward(truncated_context)
            output_logits = output_logits[:, -1, :]
            output_probs = torch.softmax(output_logits, dim=-1)
            next_token = torch.multinomial(output_probs, num_samples=1)
            generated_sequence = torch.cat((generated_sequence, next_token), dim=-1)

        return generated_sequence

        ### ========= TODO : END ========= ###



class MiniGPT_with_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Initialize the MiniGPT model with an encoder layer.
        """
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.prehead_norm = LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)
        self.apply(self._init_weights)

    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = self.pos[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
        x = self.vocab_embedding(x) + self.positional_embedding(pos)
        x = self.embed_dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        x = self.prehead_norm(x)
        x = self.head(x)

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        device = next(self.parameters()).device
        context = context.unsqueeze(0).to(device)

        for _ in range(max_new_tokens):
            logits = self.forward(context)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            next_token_tensor = torch.tensor([[next_token]], device=device)
            context = torch.cat((context, next_token_tensor), dim=1)

        return context.squeeze(0)
    
    def load_state_dict(self, state_dict, strict=True):
        state_dict = {k.replace('transformer_layers', 'encoder_layers'): v for k, v in state_dict.items()}
        state_dict = {k.replace('encoder_layers', 'decoder_layers'): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        for name, param in self.decoder_layers.named_parameters():
            if name not in state_dict:
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            raise RuntimeError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")


class Encoderlayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim)

    def forward(self, x):
        x1 = self.norm1(x)
        attention_output = self.attention(x1)
        x2 = x + attention_output  # Residual connection

        x3 = self.norm2(x2)
        feedforward_output = self.feedforward(x3)
        x4 = x2 + feedforward_output  # Residual connection

        return x4


class MiniGPT_with_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Initialize the MiniGPT model with an encoder layer.
        """
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.prehead_norm = LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)
        self.apply(self._init_weights)

    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = self.pos[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
        x = self.vocab_embedding(x) + self.positional_embedding(pos)
        x = self.embed_dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        x = self.prehead_norm(x)
        x = self.head(x)

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        device = next(self.parameters()).device
        context = context.unsqueeze(0).to(device)

        for _ in range(max_new_tokens):
            logits = self.forward(context)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            next_token_tensor = torch.tensor([[next_token]], device=device)
            context = torch.cat((context, next_token_tensor), dim=1)

        return context.squeeze(0)