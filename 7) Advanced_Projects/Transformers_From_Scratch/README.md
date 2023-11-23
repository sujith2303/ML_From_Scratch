class Preprocess_Data:
    def __init__(self,filepath,input,output,input_max_sentence_len,output_max_sequence_len):
        self.filepath                                 = filepath
        self.input                                    = input
        self.output                                   = output
        self.input_max_sentence_len                   = input_max_sentence_len
        self.output_max_sequence_len                  = output_max_sequence_len
        self.input_sentence,self.output_sentence      = self.read_data()
        self.input_sentence,self.input_tokenizer      = self.tokenize(self.input_sentence)
        self.output_sentence,self.output_tokenizer    = self.tokenize(self.output_sentence)
        self.input_sentence                           = self.pad(self.input_sentence,self.input_max_sentence_len)
        self.output_sentence                          = self.pad(self.output_sentence,self.output_max_sequence_len)
    
    def read_data(self):
        df = pd.read_csv(self.filepath)
        df = df.sample(frac = 1).reset_index(drop = True)
        input_sentence  = "<sos>" + df[self.input] +"<eos>"
        output_sentence = "<sos>" + df[self.output] +"<eos>"
        return input_sentence,output_sentence

    
    def tokenize(self,sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        return tokenizer.texts_to_sequences(sentences),tokenizer
    
    def pad(self,sentence,max_sentence_len=None):
        if max_sentence_len==None:
            max_sentence_len= max([len(i) for i in sentence])
        sentence = pad_sequences(sentence,maxlen = max_sentence_len,padding = 'post')
        return sentence



class positional_encoding(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,**kwargs):
        super().__init__(**kwargs)
        
        self.pos=np.arange(max_sentence_len).reshape(1,-1).T
        self.i=np.arange(embedding_size/2).reshape(1,-1)
        self.pos_emb=np.empty((1,max_sentence_len,embedding_size))
        self.pos_emb[:,:,0 : :2]=np.sin(self.pos / np.power(10000, (2 * self.i / embedding_size)))
        self.pos_emb[:,:,1 : :2]=np.cos(self.pos / np.power(10000, (2 * self.i / embedding_size)))
        self.positional_embedding = tf.cast(self.pos_emb,dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.positional_embedding

class paddding_mask(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self,inputs):
        mask=1-tf.cast(tf.math.equal(inputs,0),tf.float32)
        return mask[:, tf.newaxis, :] 

class create_look_ahead_mask(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self,sequence_length):
        mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
        return mask 

class input_layer_encoder(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,vocab_size,**kwargs):
        super().__init__(**kwargs)
        self.paddding_mask=paddding_mask()
        
        self.embedding=tf.keras.layers.Embedding(vocab_size,
                                                 embedding_size,
                                                 input_length=max_sentence_len,
                                                 input_shape=(max_sentence_len,))
        
        self.positional_encoding=positional_encoding(max_sentence_len,embedding_size)
    def call(self,inputs):
        mask=self.paddding_mask(inputs)
        
        emb=self.embedding(inputs)
        
        emb=self.positional_encoding(emb)
        return emb,maskc

class Encoder_layer(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 heads_num,
                 dense_num,
                 dropout_rate=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        
        self.multi_attention=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )
        
        self.Dropout=tf.keras.layers.Dropout(dropout_rate)
        
        self.ff=tf.keras.Sequential([
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(embedding_size,activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        self.add=tf.keras.layers.Add()
        
        self.norm1=tf.keras.layers.LayerNormalization()
        self.norm2=tf.keras.layers.LayerNormalization()
    def call(self,inputs,mask,training):
        
        mha=self.multi_attention(inputs,inputs,inputs,mask)
        
        norm=self.norm1(self.add([inputs,mha]))
        
        fc=self.ff(norm)
        
        A=self.Dropout(fc,training=training)
        
        output=self.norm2(self.add([A,norm]))
        
        return output

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 max_sentence_len,
                 embedding_size,
                 vocab_size,
                 heads_num,
                 dense_num,
                 num_of_encoders,
                 **kwargs):
        super().__init__(**kwargs)
        self.add=tf.keras.layers.Add()
        self.input_layer=input_layer_encoder(max_sentence_len,embedding_size,vocab_size)
        self.encoder_layer=[Encoder_layer(embedding_size,heads_num, dense_num) for i in range (num_of_encoders)]
        self.num_layers=num_of_encoders
    def call(self,inputs,training):
        emb,mask=self.input_layer(inputs)
        skip=emb
        for layer in self.encoder_layer:
            emb = layer(emb, mask,training)
            emb = self.add([skip,emb])
            skip = emb
        return emb,mask

class input_layer_decoder(tf.keras.layers.Layer):
    def __init__(self,max_sentence_len,embedding_size,vocab_size,**kwargs):
        super().__init__(**kwargs)
        self.paddding_mask=paddding_mask()
        
        self.embedding=tf.keras.layers.Embedding(vocab_size,
                                                 embedding_size,
                                                 input_length=max_sentence_len,
                                                 input_shape=(max_sentence_len,))
        
        self.positional_encoding=positional_encoding(max_sentence_len,embedding_size)
        
        self.look_ahead_mask=create_look_ahead_mask()
        self.max_sentence_len=max_sentence_len
    def call(self,inputs):
        mask=self.paddding_mask(inputs)
        
        emb=self.embedding(inputs)
        
        emb=self.positional_encoding(emb)
        
        look_head_mak=self.look_ahead_mask(self.max_sentence_len)
        look_head_mak=tf.bitwise.bitwise_and(tf.cast(look_head_mak,dtype=np.int8),tf.cast(mask,dtype=np.int8))
        return emb,look_head_mak

class decoder_layer(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 heads_num,
                 dense_num,
                 dropout_rate=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
            
        self.masked_mha=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )
        
        
        self.multi_attention=tf.keras.layers.MultiHeadAttention(
                num_heads=heads_num,
                key_dim=embedding_size,
                dropout=dropout_rate,
            )
        
        self.ff=tf.keras.Sequential([
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(dense_num,activation="relu"),
            tf.keras.layers.Dense(embedding_size,activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        self.Dropout=tf.keras.layers.Dropout(dropout_rate)
        self.add=tf.keras.layers.Add()
        
        self.norm1=tf.keras.layers.LayerNormalization()
        self.norm2=tf.keras.layers.LayerNormalization()
        self.norm3=tf.keras.layers.LayerNormalization()
        
    def call(self,inputs,encoder_output,enc_mask,look_head_mask,training):
        
        mha_out,atten_score=self.masked_mha(inputs,inputs,inputs,look_head_mask,return_attention_scores=True)
        
        Q1=self.norm1(self.add([inputs,mha_out]))
        
        mha_out2,atten_score2=self.multi_attention(Q1,encoder_output,encoder_output,enc_mask,return_attention_scores=True)
        
        Z=self.norm2(self.add([Q1,mha_out2]))
        
        fc =  self.ff(Z)
        
        A=self.Dropout(fc,training=training)
        
        output=self.norm3(self.add([A,Z]))
        return output
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 max_sentence_len,
                 embedding_size,
                 vocab_size,
                 heads_num,
                 dense_num,
                 num_of_decoders,
                 **kwargs):
        super().__init__(**kwargs)
        self.add=tf.keras.layers.Add()
        self.input_layer=input_layer_decoder(max_sentence_len,embedding_size,vocab_size)
        self.decoder_layer=[decoder_layer(embedding_size,heads_num, dense_num) for i in range (num_of_decoders)]
        self.num_layers=num_of_decoders
    def call(self,inputs,encoder_output,enc_mask,training):
        emb,look_head_mask=self.input_layer(inputs)
        skip=emb
        for layer in self.decoder_layer:
            emb = layer(emb,encoder_output,enc_mask,look_head_mask,training)
            emb = self.add([skip,emb])
            skip = emb
        return emb

class transformer(tf.keras.Model):
    def __init__(self,
                 max_sentence_len_1=None,max_sentence_len_2=None,embedding_size=None,vocab_size1=None,vocab_size2=None,
                         heads_num=None,dense_num=None,num_of_encoders_decoders=None):

        super(transformer,self).__init__()

        self.Encoder=Encoder(max_sentence_len_1,embedding_size,vocab_size1,heads_num,dense_num,num_of_encoders_decoders)
        self.Decoder=Decoder(max_sentence_len_2,embedding_size,vocab_size2,heads_num,dense_num,num_of_encoders_decoders)
        self.Final_layer=tf.keras.layers.Dense(vocab_size2, activation='relu')
        self.softmax=tf.keras.layers.Softmax(axis=-1)
    def call(self, inputs):
        input_sentence,output_sentence=inputs
        enc_output,enc_mask=self.Encoder(input_sentence)

        dec_output=self.Decoder(output_sentence,enc_output,enc_mask)

        final_out=self.Final_layer(dec_output)

        softmax_out=self.softmax(final_out)
        return softmax_out



