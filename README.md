# Disaster-Response-Pipeline-DSND

The disater response project is used Natural Language Processing and Machine Learning to Classify the messages.

Steps of the project  as follows:
<ol>
  <li>ELT Pipeline
        <ul>
           <li>Load data</li>
           <li>Clean data and save it into sqlite</li>
         </ul>
  </li>
  <li>Build ML pipeline
          <ul>
          <li>Load data</li>
          <li>Tokenize</li>
          <li>Build pipeline</li>
          <li>Train the model</li>
          <li>Evaluate the model</li>
          <li>Save the model as pkl</li>
         </ul>
  
  </li>
       
  
  <li>Build web app
     <ul>
       <li>templates files (pages of web app)
           <ul>
              <li>master page: homepage and contains on visualization charts, button and input for classifying a message</li>
              <li>go page:classify message</li> 
          </ul>
      </li>
      <li>run script file: contains on backend code for visualization charts  and classifying a message</li>
      </ul>
    </li> 
</ol>

You can run every script for these steps as follows,but before that please check you are running the scripts in the root folder of the project
<ol>
  <li><a href="https://github.com/WejdanAljadani/Disaster-Response-Pipeline-DSND/blob/master/Data/process_data.py">run the ELT pipeline script</a>:python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  </li>
  <li><a href="https://github.com/WejdanAljadani/Disaster-Response-Pipeline-DSND/blob/master/models/train_classifier.py" >run the ML pipeline script</a>:python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</li>
  <li><a href="https://github.com/WejdanAljadani/Disaster-Response-Pipeline-DSND/blob/master/app/run.py" >run the web app script</a> :app/run.py</li>
</ol>
