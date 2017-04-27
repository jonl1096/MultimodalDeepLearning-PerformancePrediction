library(RSQLite)
library(msm)
library(xtable)
library(data.table)
library(dplyr)
library(tidyr)
library(dtplyr)

db <- src_sqlite('~/Documents/Github/DB/pitchRx_14_16.sqlite3')
target_team = "Orioles"
target_team_quote = "'Orioles'"
#dbListTables(db$con) -> "action" "atbat"  "coach"  "game"   "hip"    "media"  "pitch"  "player" "po" "runner"

dbListFields(db$con, 'game')
games <- dbGetQuery(db$con, 'SELECT * FROM game')
qry <- paste('SELECT id, original_date, ampm, home_team_name, away_team_name, game_type, 
             away_win, away_loss, away_team_runs, away_team_hits, away_team_errors, away_team_hr, away_team_sb, away_team_so,
             home_win, home_loss, home_team_runs, home_team_hits, home_team_errors, home_team_hr, home_team_sb, home_team_so
             FROM game WHERE home_team_name==', target_team_quote, 'OR away_team_name==', target_team_quote, sep="")
data <- dbGetQuery(db$con, qry)

randomized_data <- data[sample(nrow(data)),]
data_len <- length(randomized_data$id)
split_size <- ceiling((3/4)*data_len)
train_data <- randomized_data[1:split_size, ]
test_data <- randomized_data[(split_size+1):data_len, ]

write.csv(train_data, file=paste(target_team,"_train.csv", sep=""), row.names = FALSE)
write.csv(test_data, file=paste(target_team,"_test.csv", sep=""), row.names = FALSE)