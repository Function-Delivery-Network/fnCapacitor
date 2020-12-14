import os
import json
import yaml
import time
from subprocess import Popen, PIPE
import datetime as dt
import requests
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from pylab import rcParams
import telegram
import numpy as np
import seaborn as sns
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class Deployment:

    def __init__(self, values_file):
        self.value = values_file
        self.cwd = os.getcwd()
        self.output_dir = os.path.join(self.cwd, "OutputtoTelegram")
        self.terraform_dir = os.path.join(self.cwd, "terraform")
        self.automation_dir = os.path.join(self.cwd, "automation")
        self.grafana_dir = os.path.join(self.cwd, "grafana")
        self.k6_dir = os.path.join(self.cwd, "k6")
        self.k3_dir = os.path.join(self.cwd, "k3s")
        # self.df = pd.DataFrame()

        self.datafile = open(os.path.join(self.automation_dir, self.value), "r")
        # self.datastore = json.loads(self.datafile.read())
        self.datastore = yaml.load(self.datafile, Loader=yaml.FullLoader)
        # print(self.datastore)
        for instance in self.datastore["model_functions"]:
            self.instance_name = instance
            self.instance = self.datastore["model_functions"][instance]
            if (self.instance["pre_test"]["cluster_deployment"]):
                self.update_tfvars_file()
                self.cluster_deployment()
            self.master_ip = self.datastore["master_ip"]
            if (self.instance["pre_test"]["function_deployment"]):
                self.function_deployment()
                time.sleep(120)
                self.k6_run()
                self.delete_function()
            if (self.instance["post_test"]["data_extraction"]):
                self.query()
            if (self.instance["post_test"]["plot"]):
                self.plot()
            #if (self.instance["post_test"]["modeling"]):
            #    self.model()
            if (self.instance["pre_test"]["cluster_deployment"]):
                self.destroy_cluster()
        self.mergeallcsv()
        if (self.datastore["model"]["generate"]):
            self.model()
        # Uncomment below line to send output to Telegram. Please update bot details
        #self.telegram_send()

    def update_tfvars_file(self):
        if (self.instance["k8"]):
            var_file = open(os.path.join(self.terraform_dir, 'var.tfvars'),"r")
        else:
            var_file = open(os.path.join(self.k3_dir, 'var.tfvars'),"r")
        var_file_content = ""
        for line in var_file:
            line = line.split("=")
            line[1] = self.instance["clusterconfig"][line[0]]
            line = '='.join(str(item) for item in line)
            var_file_content += line + '\n'
        var_file.close()
        if (self.instance["k8"]):
            var_file = open(os.path.join(self.terraform_dir, 'var.tfvars'),"w+")
        else:
            var_file = open(os.path.join(self.k3_dir, 'var.tfvars'),"w+")
        var_file.write(var_file_content)
        var_file.close()

    def cluster_deployment(self):
        if (self.instance["k8"]):
            os.chdir(self.terraform_dir)
        else:
            os.chdir(self.k3_dir)
        os.system("rm -rf terraform.* && rm -rf .terraform/")
        os.system("terraform init")
        os.system("terraform apply -var-file=var.tfvars -auto-approve && terraform output -json | jq 'with_entries(.value |= .value)' > ../config.json")
        os.system("cp var.tfvars ../var_old.tfvars")
        os.chdir(self.grafana_dir)
        os.system("rm -rf terraform.* && rm -rf .terraform/")
        os.system("terraform init && terraform apply --auto-approve")
        os.chdir(self.cwd)
        # print(cwd)
        configfile = open(os.path.join(self.cwd, 'config.json'),"r")
        configstore = json.loads(configfile.read())
        self.datastore["master_ip"] = configstore["master_ip"]
        self.write_to_json()

    def destroy_cluster(self):
        time.sleep(120)
        if (self.instance["k8"]):
            os.chdir(self.terraform_dir)
        else:
            os.chdir(self.k3_dir)
        os.system("terraform destroy -var-file=../var_old.tfvars -auto-approve")
        time.sleep(300)
    
    def ssh(self, cmd):
        try:
            ssh = Popen(["ssh", "-i", "~/.ssh/id_rsa", "-o", "StrictHostKeyChecking=no", '{}@{}'.format("ubuntu",self.master_ip), cmd], shell=False,
                            stdout=PIPE,
                            stderr=PIPE)
            result = ssh.stdout.readlines()
            if result == []:
                error = ssh.stderr.readlines()
                print ( "ERROR: %s" % error)
            else:
                print(cmd)
        except Exception as e:
            print(e)

    def function_deployment(self):
        function_deployment = "faas-cli "
        function = self.instance["function"]
        print(function["name"])
        if (function["store"] == True):
            function_deployment += "store deploy " + function["name"]
        else:
            function_deployment += "deploy --image=" + function["image"] + " --name=" + function["name"]
        cmd = "faas-cli login --tls-no-verify --username admin --password $(cat password.txt) --gateway http://127.0.0.1:31112"
        self.ssh(cmd)
        if (function["name"] == "mydb"):
            self.ssh("cd mysql-function-openfaas/ && kubectl create secret generic secret-mysql-key --from-file=secret-mysql-key=$HOME/secrets/secret_mysql_key.txt --namespace openfaas-fn && faas-cli template pull")
            function_deployment = "cd mysql-function-openfaas/ && sudo faas deploy"
        if (function["name"] == "miniodbmysql" or function["name"] == "queryfunction"):
            self.ssh("cd mysql-side/ && kubectl create secret generic secret-mysql-key --from-file=secret-mysql-key=$HOME/secrets/secret_mysql_key.txt --namespace openfaas-fn && faas-cli template pull")
            function_deployment = "cd mysql-side/ && sudo faas deploy"
            if (function["name"] == "queryfunction"):
                for label in function["openfaas"]:
                    function_deployment += " --label '" + label + "=" + str(function["openfaas"][label]) + "'"
                cmd = function_deployment
                self.ssh(cmd)
        if (function["name"] == "miniodb" or function["name"] == "queryfunction"):
            function_deployment = "cd minio/ && faas-cli template pull && sudo faas deploy"
        function_deployment += " --gateway http://127.0.0.1:31112"
        for label in function["openfaas"]:
            function_deployment += " --label '" + label + "=" + str(function["openfaas"][label]) + "'"
        cmd = function_deployment
        self.ssh(cmd)
        if (function["name"] == "miniodbmysql"):
            function_deployment = "sh scripts/miniodbmysql.sh && cd miniodbmysql/ && sudo faas deploy"
            for label in function["openfaas"]:
                function_deployment += " --label '" + label + "=" + str(function["openfaas"][label]) + "'"
            cmd = function_deployment
            self.ssh(cmd)
        if (function["name"] == "queryfunction"):
            function_deployment = "sh scripts/queryfunction.sh && cd queryfunction/ && sudo faas deploy"
            for label in function["openfaas"]:
                function_deployment += " --label '" + label + "=" + str(function["openfaas"][label]) + "'"
            cmd = function_deployment
            self.ssh(cmd)

    def delete_function(self):
        function = self.instance["function"]
        cmd = "faas-cli remove " + function["name"] + " --gateway http://127.0.0.1:31112"
        self.ssh(cmd)
        if (function["name"]=="miniodbmysql"):
            cmd = "faas-cli remove mysqlside --gateway http://127.0.0.1:31112"
            self.ssh(cmd)
        os.system("sleep 10")

    def k6_run(self):
        os.chdir(self.k6_dir)
        #print(instance)
        function = self.instance["function"]["name"]
        payload = self.instance["test"]["function_params"]
        file_name = self.instance["test"]["test_run"]
        k6 ="MASTER_IP=" + self.master_ip + " PAYLOAD=" + payload + " FUNCTION=" + function +  \
            " k6 run " + file_name + " --out influxdb=http://"  \
            + str(self.datastore["database"]["host"])  \
            + ":" + str(self.datastore["database"]["port"]) +"/" + self.datastore["database"]["name"]
        print(k6)
        for options in self.instance["test"]["k6"]:
            if(options == "customized"):
                for m in range(1, self.instance["test"]["k6"][options]["stage"]+1):
                    m = m*3
                    k6 += " --stage" + " 1m:" + str(m)
            elif(type(self.instance["test"]["k6"][options]) == list):
                for i in self.instance["test"]["k6"][options]:
                    k6 += " --" + options + " " +str(i)
            else:
                k6 += " --" + options + " " +str(self.instance["test"]["k6"][options])
        start_time = dt.datetime.now(tz=dt.timezone.utc)
        start_time = start_time.replace(tzinfo=dt.timezone.utc).timestamp()
        #print(k6)
        os.system(k6)
        end_time = dt.datetime.now(tz=dt.timezone.utc)
        end_time = end_time.replace(tzinfo=dt.timezone.utc).timestamp()
        os.chdir(self.cwd)
        self.instance["time"]["start"]= str(start_time)
        self.instance["time"]["end"] = str(end_time)
        self.write_to_json()

    def query(self):
        self.df = pd.DataFrame()
        #print(self.datastore["query"])
        self.datastore["query"]["cpu"]["formatter"] = self.instance["clusterconfig"]["core"]
        self.datastore["query"]["mem"]["formatter"] = 1024*1024*1024*self.instance["clusterconfig"]["memory"]
        self.write_to_json()
        for i in self.datastore["query"]:
            if self.datastore["query"][i]["query_split"] == False:
                url = "http://"+ str(self.datastore["prometheus"]["host"])+ ":" + str(self.datastore["prometheus"]["port"]) + "/" \
                    + str(self.datastore["prometheus"]["api"]) + str(self.datastore["query"][i]["query"]) + "&start=" \
                        + str(self.instance["time"]["start"]) \
                        + "&end=" + str(self.instance["time"]["end"]) \
                        + "&step=" + str(self.datastore["query"][i]["step"])
            else:
                url = "http://"+ str(self.datastore["prometheus"]["host"])+ ":" + str(self.datastore["prometheus"]["port"]) + "/" \
                    + str(self.datastore["prometheus"]["api"]) + str(self.datastore["query"][i]["query"]) \
                        + self.instance["function"]["name"] + str(self.datastore["query"][i]["query1"]) \
                        + str(self.datastore["query"][i]["formatter"]) + "*100&start=" \
                        + str(self.instance["time"]["start"]) \
                        + "&end=" + str(self.instance["time"]["end"]) \
                        + "&step=" + str(self.datastore["query"][i]["step"])
            print(url)
            receive = requests.get(url)
            # print(receive.json())
            self.data_formatter(receive.json(), self.datastore["query"][i]["name"])
        filename = self.instance["function"]["name"] + self.instance["time"]["start"] + "_" + self.instance["time"]["end"] + ".csv"
        if(self.instance["training"] is True):
            self.df.to_csv(os.path.join(self.output_dir, "training",filename), index=True)
        else:
            self.df.to_csv(os.path.join(self.output_dir, "testing", filename), index=True)

    def data_formatter(self, result, column):
        # print(result["data"]["result"][0]["values"])
        self.time = []
        data = []
        try:
            for value in result["data"]["result"][0]["values"]:
                datetime_object = dt.datetime.fromtimestamp(value[0])
                self.time.append(datetime_object)
                data.append(float(value[1]))
                #print(self.time,data)
            self.toTable(column, data)
        except Exception as e:
            print(e)

    def toTable(self, column, data):
        if self.df.empty:
            tableData = {'Time': pd.Series(self.time), column: pd.Series(data)}
            self.df = pd.DataFrame(tableData)
            self.df.set_index("Time", inplace = True)
            self.length = len(self.df.cpu.values)
        else:
            df2 = pd.DataFrame({'Time': pd.Series(self.time), 'Value': pd.Series(data)})
            df2.set_index("Time", inplace = True)
            #print(df2.Value.values)
            if (self.length < len(df2.Value.values)):
                df2.drop(df2.index[self.length:], inplace=True)
            print(len(df2.Value.values),self.length)
            if (len(df2.Value.values) < self.length):
                for i in range(len(df2.Value.values), self.length):
                    df_new = pd.DataFrame({'Time': [dt.datetime.now()], 'Value': [1]})
                    df_new.set_index("Time", inplace = True)
                    df2 = df2.append(df_new, ignore_index = True)
            self.df[column] = df2.Value.values

    def plot(self):
        rcParams['figure.figsize'] = 30, 20
        rcParams["legend.loc"] = 'upper left'
        rcParams['axes.labelsize'] = 16
        rcParams['axes.titlesize'] = 20
        rcParams["font.size"] = 16
        ax = []
        fig = plt.subplots()
        #print(self.df.columns)
        filename = self.instance["function"]["name"] + self.instance["time"]["start"] + "_" + self.instance["time"]["end"] + ".csv"
        if(self.instance["training"] is True):
            self.df = pd.read_csv(os.path.join(self.output_dir, "training", filename))
        else:
            self.df = pd.read_csv(os.path.join(self.output_dir, "testing", filename))
        #self.df.set_index("Time", inplace = True)
        if len(self.df.columns) <= 4:
            fig, axs = plt.subplots(2, 2)
        elif len(self.df.columns) <= 9:
            fig, axs = plt.subplots(3, 3)
            plot_array = "33"
        else:
            fig, axs = plt.subplots(4, 4)
            plot_array = "44"
        for col, ax in zip(range(1,len(self.df.columns)), axs.flatten()):
            ax.plot(self.df.index,self.df[self.df.columns[col]])
            ax.set_title(self.df.columns[col])
        fig.delaxes(axs[3][2])
        fig.delaxes(axs[3][3])
        filename = "updated" + self.instance["function"]["name"] + self.instance["time"]["start"] + "_" + self.instance["time"]["end"] + ".png"
        #plt.show()
        if(self.instance["training"] is True):
            plt.savefig(os.path.join(self.output_dir, "training", filename))
        else:
            plt.savefig(os.path.join(self.output_dir, "testing", filename))

    def model(self):
        df = pd.read_csv(os.path.join(self.output_dir, "training", "combined_" + self.instance["function"]["name"] + "training.csv"))
        df_test = pd.read_csv(os.path.join(self.output_dir, "testing" , "combined_" + self.instance["function"]["name"] + "testing.csv"))
        df = df.drop(df[df.responsetime > self.datastore["model"]["sla"]].index)
        df.pop('Time')
        df_test = df_test.drop(df_test[df_test.responsetime > self.datastore["model"]["sla"]].index)
        df_test.pop('Time')
        df = df.drop(df[df.totalcpu > 16].index)
        df_t = pd.DataFrame()
        df_ttest = pd.DataFrame()
        # training data
        df_t['total_cpu_util'] = (df['totalcpuUtilization']*(df['totalcpu']*0.67))/100                    
        df_t['total_mem_util'] = (df['totalmemoryUtilization']*df['totalmemory'])*1e-9
        df_t['responsetime'] = df['responsetime']
        df_t['requests'] = df['requests']
        # testing data
        df_ttest['total_cpu_util'] = (df_test['totalcpuUtilization']*(df_test['totalcpu']*0.67))/100
        df_ttest['total_mem_util'] = (df_test['totalmemoryUtilization']*df_test['totalmemory'])*1e-9
        df_ttest['responsetime'] = df_test['responsetime']
        df_ttest['requests'] = df_test['requests']
        df_t = pd.get_dummies(df_t, prefix='', prefix_sep='')
        i = 0
        test_dataset = df_ttest
        test_features = test_dataset.copy()
        test_labels = test_features.pop('requests')
        loss, score, model = [], [], []
        # k fold
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.datastore["model"]["patience"])
        X = df_t.loc[:,['total_cpu_util','total_mem_util', 'responsetime']].values
        y = df_t.loc[:,['requests']].values
        kf = KFold(n_splits=6, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            train_features, val_features = X[train_index], X[test_index]
            train_labels, val_labels = y[train_index], y[test_index]

        # Divide dataset
        #train_dataset = df_t.sample(frac=0.8, random_state=0)
        #val_dataset = df_t.drop(train_dataset.index)
        #test_dataset = df_ttest

        #train_features = train_dataset.copy()
        #val_features = val_dataset.copy()
        #test_features = test_dataset.copy()

        # Create labels
        #train_labels = train_features.pop('requests')
        #val_labels = val_features.pop('requests')
        #test_labels = test_features.pop('requests')

        # Normalization
            input = np.array(train_features)
            input_normalizer = preprocessing.Normalization(input_shape=[3,])
            input_normalizer.adapt(input)

        # Create Model
            dnn_model = None
            dnn_model = self.build_and_compile_model(input_normalizer)

            print(dnn_model.summary())
            self.modelname = 'dnn_model_'+str(i)

            history = dnn_model.fit(
                train_features, train_labels,
                validation_split=0.2,
                verbose=0, epochs=3000,callbacks=[callback])
            #print(history)
            self.plot_loss(history)
            loss.append(dnn_model.evaluate(val_features, val_labels,verbose=0))
            ## Make Predictions
            test_predictions = dnn_model.predict(test_features).flatten()

            self.plot_prediction(test_labels, test_predictions)
            R = r2_score(test_labels, test_predictions)*100
            model.append('dnn_model_'+str(i))
            score.append(R)
            i = i+1
            #test_results['dnn_model_'+str(i)] = [dnn_model.evaluate(
            #    val_features, val_labels,
            #    verbose=0), R]
        model = np.array(model)
        score = np.array(score)
        loss = np.array(loss)
        data = np.array([model, loss, score]).T
        data = pd.DataFrame(data,columns=["model", "loss", "score"])
        filename = "model" +self.instance["function"]["name"]  + ".csv"
        data.to_csv(os.path.join(self.output_dir, "model",filename), index=True)
    
    def plot_loss(self, history):
        plot_loss = plt.axes()
        plot_loss.plot(history.history['loss'], label='loss')
        plot_loss.plot(history.history['val_loss'], label='val_loss')
        plot_loss.set_ylim([0, max(history.history['loss']+history.history['val_loss'])+100])
        plot_loss.set_xlabel('Epoch')
        plot_loss.set_ylabel('Error [requests]')
        plot_loss.legend()
        plot_loss.grid(True)
        plt.savefig(os.path.join(self.output_dir, "model", self.modelname+"loss.png"))

    def plot_prediction(self, test_labels, test_predictions):
        plot_prediction = plt.axes(aspect='equal')
        #a = plot_prediction.axes(aspect='equal')
        plot_prediction.scatter(test_labels, test_predictions)
        plot_prediction.set_xlabel('True Values requests')
        plot_prediction.set_ylabel('Predictions requests')
        lims = [0, max(list(test_labels)+list(test_predictions))+100]
        plot_prediction.set_xlim(lims)
        plot_prediction.set_ylim(lims)
        #_ = plot_prediction.plot(lims, lims)
        plt.savefig(os.path.join(self.output_dir, "model", self.modelname+"prediction.png"))

    def build_and_compile_model(self, norm):
        model = None
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu',name="dense_one"),
            layers.Dense(64, activation='relu',name="dense_two"),
            layers.Dense(1,name="dense_three")
        ])

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def write_to_json(self):
        os.remove(os.path.join(self.automation_dir, self.value))
        self.datafile = open(os.path.join(self.automation_dir, self.value), "w")
        yaml.dump(self.datastore, self.datafile, indent=4)
        self.datafile.close()

    def mergeallcsv(self):
        dir = ["testing", "training"]
        for dir_name in dir:
            dir_list= []
            for p,n,f in os.walk(os.path.join(self.output_dir, dir_name)):
                for a in f:
                    a = str(a)
                    if a.endswith('.csv'):
                        dir_list.append(os.path.join(p,a))
            combined_csv = pd.concat([pd.read_csv(f) for f in dir_list ])
            filename= "combined_" + self.instance["function"]["name"] + dir_name + ".csv"
            filepath=os.path.join(self.output_dir, dir_name, filename)
            combined_csv.to_csv( filepath, index=False, encoding='utf-8-sig')

    def telegram_send(self, chat_id="yourBotChatID", token='yourBotToken'):
        bot = telegram.Bot(token=token)
        filename = self.instance_name + self.instance["function"]["name"] + self.instance["time"]["start"] + "_" + self.instance["time"]["end"] + ".zip"
        copy = "cp " + os.path.join(self.automation_dir, "values.yaml") + " " +self.output_dir
        os.system(copy)
        compress = "tar -zcvf " + filename + " " +self.output_dir
        os.chdir(self.cwd)
        os.system(compress)
        bot.send_document(chat_id=chat_id, document=open(os.path.join(self.cwd, filename), 'rb'))
        #for file in os.listdir(self.output_dir):
        #    if '.png' in file:
        #        bot.send_photo(chat_id=chat_id, photo=open(os.path.join(self.output_dir, file), 'rb'))
        #    else:
        #        bot.sendDocument(chat_id=chat_id, document=open(os.path.join(self.output_dir, file), 'rb'))
        os.system("rm -rf OutputtoTelegram/testing/* && rm -rf OutputtoTelegram/training/* && rm -rf OutputtoTelegram/model/*")
        delete_tar = "rm -rf " + filename
        os.system(delete_tar)

if __name__ == '__main__':
    Deployment('values.yaml')
