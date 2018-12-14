# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import os
import statistics

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(training.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.AdamOptimizer()

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
  
  
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()
  
  
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

    
# List of MAE's for top 11 players
maes = []
overall_maes = []
error_list = []

# Iterate through the contents of Data folder
players = os.listdir("./Data/")
for player in players:
    # Currently configured to run 1 player at a time.
    # To run every player in 1 execution, remove break at end of loop
    # and comment out the line below
    player = "James Harden"
    print(player)    
    for _i in range(0,10):
        print(player + ": " + str(_i))
        dir_var = "./Data/" + player + "/"

        # Open up 16-17 and 17-18 and store as training data
        training = pd.read_csv(dir_var + "training_all_players.csv")
        training = training.drop(['FG_PCT_SEASON', 'FG3_PCT_SEASON', 'FT_PCT_SEASON', 'TEAM_PACE', 'OPP_PACE', 'PF', 'ACTUAL_PLAYER_USG', 'ACTUAL_PACE', 'MIN','SEASON_ID', 'Player_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS', 'VIDEO_AVAILABLE'], axis=1)
        # Leaves MIN_SEASON, OPP_DEF_RTG, FGA_SEASON, FG3A_SEASON, FTA_SEASON  
        
        
        training_labels = training['PTS'].as_matrix()
        training = training.drop('PTS', axis=1).as_matrix()
        
       
        # Open up 18-19
        testing = pd.read_csv(dir_var + "testing.csv")
        testing = testing.drop(['FG_PCT_SEASON', 'FG3_PCT_SEASON', 'FT_PCT_SEASON', 'TEAM_PACE', 'OPP_PACE', 'PF', 'ACTUAL_PLAYER_USG', 'ACTUAL_PACE', 'MIN','SEASON_ID', 'Player_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS', 'VIDEO_AVAILABLE'], axis=1)
        testing['PTS'] = (testing['PTS']).astype(int)               
        testing_labels = testing['PTS'].as_matrix()
        testing = testing.drop('PTS', axis=1).as_matrix()
        

        #print(training.shape)
        #print(testing.shape)
        #print(training)

        model = build_model()
        model.summary()

        EPOCHS = 500
        # Store training stats
        history = model.fit(training, training_labels, epochs=EPOCHS,
                            validation_split=0.1, verbose=1,
                            callbacks=[PrintDot()])

        # Plot training loss                   
        #plot_history(history)

        [loss, mae] = model.evaluate(testing, testing_labels, verbose=0)

        print("Testing set Mean Abs Error: {:7.2f}".format(mae))
        maes.append(mae)

        test_predictions = model.predict(testing).flatten()

        plt.scatter(testing_labels, test_predictions)
        plt.xlabel('True Values ')
        plt.ylabel('Predictions')
        plt.axis('on')
        #plt.xlim(0, 60)
        plt.ylim(0, 60)
        plt.show()

        errors = test_predictions - testing_labels
        for error in errors:
            error_list.append(error)
    
    overall_maes.append(statistics.mean(maes))
    
    break # Remove me if want all player execution
    
plt.hist(error_list, bins = 50)
plt.xlabel("Prediction Error ")
_ = plt.ylabel("Count")
plt.show()

print("10x Averaged for each player")
print(overall_maes)
print("Average of each player's 10x Averaged MAE")
print(statistics.mean(overall_maes))