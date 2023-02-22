from flask import Flask,request, url_for, redirect, render_template,  send_file
import pickle
import numpy as np
import pandas as pd
import collections.abc as collections
from sklearn.neighbors import DistanceMetric

from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel,pipeline
import re

import nltk
from nltk.corpus import stopwords


from nltk.stem import WordNetLemmatizer
stop_w = stopwords.words('english')


dist = DistanceMetric.get_metric('hamming')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'input files'



def remove_punc(line):
  line = re.sub('[0-9@]', '',line)
  line = re.sub('[#[]', '',line)
  line = re.sub(']', '',line)
  line = re.sub('[\'":,.?!&)(]', '',line)
  line = line.rstrip().lstrip()
  return line


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")

multi_sent = load_model('model_multi_sentimentv2_withoutCuda.h5')
sent = load_model('Trip_model_sentimentv2_withoutCuda.h5')






lemmatizer = WordNetLemmatizer()





NCF = load_model('NCF.h5')
agg = load_model('agg.h5')


sim_df = pd.read_csv('data_simple_sent.csv')
mul_df = pd.read_csv('data_multi_sent.csv')
k = 3

def ncf(user,item):
    prd = NCF.predict([np.array([user]), np.array([item])])[0]
    return np.array([min(pr,5) for pr in prd])

def cf_ssent(user,item):



    if item in sim_df.offering_id.values:
        if user in sim_df['user_id'].unique():
            precedent_data = sim_df[sim_df['user_id'] ==user]



            precedent_hotels = precedent_data.offering_id.values
            prcd_reviews = precedent_data.sentiment_score.values

            prcedent_data_sentiment_review = {}
            for h_id, rev in zip(precedent_hotels, prcd_reviews):
                prcedent_data_sentiment_review[h_id] = rev



            maybe_sim_users = []
            sim = {}


            users = sim_df[sim_df['offering_id'] == item].user_id.unique()
            for user1 in users:
                if (user1 != user):

                    user1_data = sim_df[sim_df['user_id'] == user1]
                    sim_items = set(precedent_hotels).intersection(set(user1_data.offering_id.values))
                    if len(sim_items) > 1:

                        maybe_sim_users.append(user1)

                        user_rat = []
                        user1_rat = []
                        for item in sim_items:
                            user_rat.append(prcedent_data_sentiment_review[item])

                        for item in sim_items:
                            rev = user1_data[user1_data['offering_id'] == item].sentiment_score.values

                            user1_rat.append(rev[0])

                        if (1 - dist.pairwise([user_rat], [user1_rat])[0][0]) > 0.5:
                            sim[user1] = 1 - dist.pairwise([user_rat], [user1_rat])[0][0]

            if len(sim.keys()) > 0:  # similar users
                rat = [0.0, 0.0, 0.0, 0.0, 0.0]
                sum_sim = 0.0
                for user1 in list(sim.keys())[:k]:
                    rat += sim[user1] * \
                           sim_df[(sim_df['user_id'] == user1) & (sim_df['offering_id'] == item)][
                               ['cleanliness', 'location', 'rooms', 'service', 'value']].values[0]
                    sum_sim += sim[user1]
                rat = rat / sum_sim


            else:  # no similar users
                rat = [0.0, 0.0, 0.0, 0.0, 0.0]
                for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
                    rat[i] = np.mean(sim_df[sim_df['offering_id'] == item][[col]].values)

        else:  # new_user
            rat = [0.0, 0.0, 0.0, 0.0, 0.0]
            for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
                rat[i] = np.mean(sim_df[sim_df['offering_id'] ==item][[col]].values)

    elif user in mul_df['user_id'].unique():

             # new_item
            rat = [0.0, 0.0, 0.0, 0.0, 0.0]
            for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
                rat[i] = np.mean(sim_df[[col]].values)
    else:  # new_user , new_item
        # rnd_petubation = random.uniform(-1,1)
        # val = rnd_petubation + 3.0
        rat = [sim_df['cleanliness'].mean(), sim_df['location'].mean(), sim_df['rooms'].mean()
            , sim_df['service'], sim_df['value']]


    return np.array(rat)

def cf_msent(user,item):

    if item in mul_df.offering_id.values:

        if user in mul_df['user_id'].unique():
            precedent_data = mul_df[mul_df['user_id'] == user]



            precedent_hotels = precedent_data.offering_id.values
            prcd_reviews = precedent_data[['cleanliness_sent', 'location_sent', 'rooms_sent', 'service_sent',
                                           'value_sent']].values

            prcedent_data_sentiment_review = {}
            for h_id, rev in zip(precedent_hotels, prcd_reviews):
                prcedent_data_sentiment_review[h_id] = rev



            maybe_sim_users = []
            sim = {}


            users = mul_df[mul_df['offering_id'] == item].user_id.unique()
            for user1 in users:
                if (user1 != user):

                    user1_data = mul_df[mul_df['user_id'] == user1]
                    sim_items = set(precedent_hotels).intersection(set(user1_data.offering_id.values))
                    if len(sim_items) > 1:

                        maybe_sim_users.append(user1)

                        user_rat = []
                        user1_rat = []
                        for item in sim_items:
                            for j in range(5):
                                user_rat.append(prcedent_data_sentiment_review[item][j])

                            rev = user1_data[user1_data['offering_id'] == item][
                                ['cleanliness_sent', 'location_sent', 'rooms_sent', 'service_sent',
                                 'value_sent']].values

                            for j in range(5):
                                user1_rat.append(rev[0][j])

                        if 1 - abs(dist.pairwise([user_rat], [user1_rat])[0][0]) > 0.4:
                            sim[user1] = 1 - abs(dist.pairwise([user_rat], [user1_rat])[0][0])

            if len(sim.keys()) > 1:  # similar users


                rat = [0.0, 0.0, 0.0, 0.0, 0.0]
                rat_mul = [0.0, 0.0, 0.0, 0.0, 0.0]
                sum_sim = 0.0
                # for user1 in list(sim.keys())[:k]:
                # rat += sim[user1] * train_data[(train_data['user_id']==user1) & (train_data['offering_id']==hotel_id ) ][['cleanliness','location','rooms','service','value']].values[0]
                # sum_sim+= sim[user1]
                # rat = rat / sum_sim

                for user1 in list(sim.keys())[:k]:
                    user1_data = mul_df[mul_df['user_id'] == user1]
                    rev = mul_df[(mul_df['user_id'] == user1) & (mul_df['offering_id'] == item)][
                        ['cleanliness_sent', 'location_sent', 'rooms_sent', 'service_sent',
                         'value_sent']].values[0]
                    rat_mul = mul_df[(mul_df['user_id'] == user1) & (mul_df['offering_id'] == item)][
                        ['cleanliness', 'location', 'rooms', 'service', 'value']].values[0]

                    # rat = rat+ sim[user1]*0.5*(rev*5+rat_mul)
                    rat += rat_mul * sim[user1]
                    sum_sim += sim[user1]
                rat = rat / sum_sim




            else:  # no similar users
                rat = [0.0, 0.0, 0.0, 0.0, 0.0]
                for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
                    rat[i] = np.mean(mul_df[mul_df['offering_id'] == item][[col]].values)

        else:  # new_user
            rat = [0.0, 0.0, 0.0, 0.0, 0.0]
            for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
                rat[i] = np.mean(mul_df[mul_df['offering_id'] == item][[col]].values)

    elif mul_df[mul_df['user_id'] == user].shape[0] > 0:  # new_item
        precedent_data = mul_df[mul_df['user_id'] == user]
        rat = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i, col in zip([0, 1, 2, 3, 4], ['cleanliness', 'location', 'rooms', 'service', 'value']):
            rat[i] = np.mean(precedent_data[[col]].values)
    else:  # new_user , new_item
        # rnd_petubation = random.uniform(-1,1)
        # val = rnd_petubation + 3.0
        rat = [mul_df['cleanliness'].mean(), mul_df['location'].mean(), mul_df['rooms'].mean()
            , mul_df['service'], mul_df['value']]

    return np.array(rat)


def NCF_FCBS(user,item):

    ncf_pred = ncf(user,item)
    cf_pred = cf_ssent(user,item)

    return 0.3*ncf_pred+0.7*np.array(cf_pred)
def NCF_FCBS_MA(user,item):

    ncf_pred = ncf(user,item)
    cf_pred = cf_msent(user,item)

    return 0.5*ncf_pred+0.5*np.array(cf_pred)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try :
            user_id = int(request.form['USERID'])
            items = sim_df[sim_df['user_id']!=user_id].offering_id.unique()
            print(user_id,items)

            prd = {}
            if request.form['algo'] == 'NCF':
                for item in items[:200]:
                    prd[item] = agg.predict(np.array([ncf(user_id, item)]))[0][0]
                print('NCF', prd)
            elif request.form['algo'] == 'FCBS':

                for item in items[:200]:
                    prd[item] = agg.predict(np.array([cf_ssent(user_id, item)]))[0][0]
                print('FCBS', prd)
            elif request.form['algo'] == 'FCBS-MA':

                for item in items[:200]:


                    prd[item] = agg.predict(np.array([cf_msent(user_id, item)]))[0][0]
                print('FCBS-MA', prd)
            elif request.form['algo'] == 'Hybridation1':
                for item in items[:200]:
                    prd[item] = agg.predict(np.array([NCF_FCBS(user_id, item)]))[0][0]
                print('NCF_FCBS', prd)
            elif request.form['algo'] == 'Hybridation2':
                for item in items[:200]:
                    prd[item] = agg.predict(np.array([NCF_FCBS_MA(user_id, item)]))[0][0]
                print('NCF_FCBS-MA', prd)

            to_recommend = [k for k, v in sorted(prd.items(), key=lambda item: item[1],reverse=True)][:3]
            it_info = []
            i=1
            for it in to_recommend:
                it_info.append('Hotel '+str(i)+ ' : '+ str(sim_df[sim_df['offering_id']==it]['offering_id'].values[0]))
                i+=1


            return render_template("recommendation.html",message='Recommendation pour l\'utilisateur '+str(user_id), item1=it_info[0],item2=it_info[1],item3=it_info[2])
        except:
            return render_template("recommendation.html")

    else:
        return render_template("recommendation.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message,cls,loc,rom,ser,val = '','','','','',''
        prd = []
        user_id = int(request.form['USERID'])
        item_id = int(request.form['ITEMID'])
        if request.form['algo'] == 'NCF':

            prd = ncf(user_id,item_id)
            print('NCF',prd)


        elif request.form['algo'] == 'FCBS':
            prd = cf_ssent(user_id,item_id)
            print('FCBS',prd)
        elif request.form['algo'] == 'FCBS-MA':
            prd = cf_msent(user_id,item_id)
            print('FCBS-MA',prd)
        elif request.form['algo'] == 'Hybridation1':
            prd = NCF_FCBS(user_id,item_id)
            print('NCF_FCBS',prd)
        elif request.form['algo'] == 'Hybridation2':
            prd = NCF_FCBS_MA(user_id,item_id)
            print('NCF_FCBS-MA',prd)

        prd_agg = agg.predict(np.array([prd]))[0][0]
        print(prd_agg)
        if len(prd)>0:
            message = 'Resultat de prédiction de l\'utilisateur ' + str(user_id) + ' sur l\'item ' + str(item_id) + ' est :'
            cls = 'Cleanliness :  ' + str(prd[0])
            loc = 'Location    :  ' + str(prd[1])
            rom = 'Rooms       :  ' + str(prd[2])
            ser = 'Service     :  ' + str(prd[3])
            val = 'Value       :  ' + str(prd[4])
            ag = 'Overall     :  ' + str(prd_agg)


        return render_template('predict.html',message=message,cls=cls,rom=rom,loc=loc,ser=ser,val=val,agg=ag)
    else:
        return render_template('predict.html')
import os
from werkzeug.utils import secure_filename
@app.route('/csv', methods=['GET', 'POST'])
def csv():

    if  request.method == 'POST':
        print(request.files,request.form)
        if 'file' not in request.files:
            print('nothing')
        else:
            print(request.files['file'],request.form['algo'])
            file = request.files['file']

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prd = []




            try :
                users = data.user_id.values
                items = data.offering_id.values
                print('bien')
                if request.form['algo'] == 'NCF':
                    for user,item in zip(users,items):
                        prd.append(ncf(user, item))
                    print('NCF', prd)
                elif request.form['algo'] == 'FCBS':
                    for user, item in zip(users, items):
                        prd.append(cf_ssent(user, item))
                    print('FCBS', prd)
                elif request.form['algo'] == 'FCBS-MA':
                    for user, item in zip(users, items):
                        prd.append(cf_msent(user, item))
                    print('FCBS-MA', prd)
                elif request.form['algo'] == 'Hybridation1':
                    for user, item in zip(users, items):
                        prd.append(NCF_FCBS(user, item))
                    print('Hyb1', prd)
                elif request.form['algo'] == 'Hybridation2':
                    for user, item in zip(users, items):
                        prd.append(NCF_FCBS_MA(user, item))
                    print('hyb2', prd)
                prd = np.array(prd)
                data['cleanliness'] =prd[:,0]
                data['location'] =prd[:,1]
                data['rooms']=prd[:,2]
                data['service']=prd[:,3]
                data['value']=prd[:,4]

                prd_agg = agg.predict(np.array(prd))
                data['overall'] = prd[:,0]



                data.to_csv('out/out.csv',index=False)

                return redirect(url_for('download_file_page'))

            except:

                pass



        return render_template('csv.html')
    else:
        return render_template('csv.html')

@app.route('/download_page')
def download_file_page():
    return render_template('download.html')


@app.route('/download_file', methods=['GET', 'POST'])
def download_file():
	#path = "html2pdf.pdf"
	#path = "info.xlsx"
	path = "out/out.csv"
	print('download file')
	return send_file(path, as_attachment=True)


@app.route('/SA', methods=['GET', 'POST'])
def sentiment_simple():
    if request.method == 'POST':
        comment = ''
        message = ''
        try :

            text = request.form['Comment']

            text_without_punc = remove_punc(text)



            clean_text = ' '.join([lemmatizer.lemmatize(w.lower()) for w in text_without_punc.split(' ') if
                                   lemmatizer.lemmatize(w.lower()) not in stop_w])
            if len(clean_text.split(' '))>0:
                encoded_input = tokenizer([clean_text], return_tensors='tf', max_length=150,
                                          # maximum length of a sentence  (TODO Figure the longest passage length)
                                          truncation='longest_first', padding='max_length'
                                          )

                output = model(encoded_input)


                prd = 'postive' if sent.predict(output['last_hidden_state'])[0] >= 0.5 else 'negative'
                message = 'Resultat de l\'analyse des sentiments'
                comment = 'Score : ' +prd
                print(prd)
        except:
            pass
        return render_template('SA_simple.html',comment = comment,message=message)
    else:
        return render_template('SA_simple.html')

@app.route('/SAMA', methods=['GET', 'POST'])
def sentiment_MA():
    if request.method == 'POST':
        comment = ''
        message = ''
        try:

            text = request.form['Comment']

            text_without_punc = remove_punc(text)

            clean_text = ' '.join([lemmatizer.lemmatize(w.lower()) for w in text_without_punc.split(' ') if
                                   lemmatizer.lemmatize(w.lower()) not in stop_w])
            if len(clean_text.split(' ')) > 3:
                encoded_input = tokenizer([clean_text], return_tensors='tf', max_length=150,
                                          # maximum length of a sentence  (TODO Figure the longest passage length)
                                          truncation='longest_first', padding='max_length'
                                          )

                output = model(encoded_input)


                prd = multi_sent.predict(output['last_hidden_state'])[0]

                prd = ['positive' if pr >= 0.5 else 'negative' for pr in prd]
                cln = 'Cleanliness : ' + prd[0]
                loc = 'Location : ' + prd[1]
                rom = 'Rooms : ' + prd[2]
                ser = 'Service : ' + prd[3]
                val = 'Value : ' + prd[4]

                message = 'Resultat de l\'analyse des sentiments'

                print(prd)
                return render_template('SA_MA.html', cln=cln, loc=loc, rom=rom, ser=ser, val=val, message=message)
            else:
                return render_template('SA_MA.html')
        except:
            return render_template('SA_MA.html')

    else:
        return render_template('SA_MA.html')

if __name__ == '__main__':
    app.run(debug=True)