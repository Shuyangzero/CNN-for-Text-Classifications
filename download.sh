wget http://phontron.com/data/topicclass-v1.tar.gz
tar xvf topicclass-v1.tar.gz
mv topiclass data
rm topicclass-v1.tar.gz
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip xvf glove.6B.zip
mv glove.6B/* data/