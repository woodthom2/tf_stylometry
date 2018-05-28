# TF Stylometry
Stylometry demo in TensorFlow

You need:

* Python 3 (I recommend Anaconda)
* Tensorflow 1.4+
* GenSim word vectors file GoogleNews-vectors-negative300.bin
(current links are https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and https://github.com/mmihaltz/word2vec-GoogleNews-vectors, this is a huge file so unfortunately I can't host it, please let me know if links break.)


# Instructions

This is how to run on the basic toy example of Anne, Charlotte and Emily Brontë's works which are in the folder data/raw.

You may want to put your own texts that you're interested in classifying into the folder, however I was only able to store works in the repo that are already out of copyright.

1. Download the GenSim word vectors file from one of the above links

2. Launch Jupyter Notebook

3. Open Preprocess_data_1_Determine_vocabulary.ipynb and change the absolute path to the path to your downloaded word vectors file. Run the notebook. It will write some gz files to the data folder and also write the preprocessed (tokenised) texts to data/processed.

4. Kill the Jupyter kernel if it's still running otherwise you'll run out of memory.

5. Open and run Preprocess_data_2_Convert_texts_to_token_IDs.ipynb. Again, kill the kernel at the end.

6. Open and run Train.ipynb. I suggest to run it for about 30 minutes.

7. Make a note of the last file in folder runs/checkpoints. This is your model at the point that you stopped it training.

8. Correct the path given to saver.restore inside Execute.ipynb to point to the latest model. Run Execute.ipynb. The output is an array of probabilities representing the likelihood that Anne, Charlotte and Emily Brontë wrote the given text (in alphabetical order).

```
array([[0.43884012, 0.35928553, 0.20187436]], dtype=float32)
```

9. To run as a webserver edit author_inference.py to point to the correct model and run

```
python webserver.py
```
 
and go to localhost:5000 in your browser.

# Acknowledgement

I've taken the demo training data from https://github.com/mikekestemont/pystyl, originally from the Gutenberg Project

I based the text classification CNN on https://github.com/dennybritz/cnn-text-classification-tf.