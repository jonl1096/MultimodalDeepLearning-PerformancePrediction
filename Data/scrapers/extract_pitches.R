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
#games <- dbGetQuery(db$con, 'SELECT * FROM game')
qry <- paste('SELECT original_date, home_team_name, away_team_name,
             away_win, away_loss, away_team_hits, away_team_errors, away_team_hr, away_team_sb, away_team_so,
             home_win, home_loss, home_team_hits, home_team_errors, home_team_hr, home_team_sb, home_team_so,
             away_team_runs, home_team_runs, gameday_link
             FROM game WHERE home_team_name==', target_team_quote, 'OR away_team_name==', target_team_quote, sep="")

data <- dbGetQuery(db$con, qry)

# acquire isHome
data$isHome <- -1
data$isHome[data$home_team_name == target_team] <- 1
data$score_diff <- 0
data$score_diff <- (data$home_team_runs - data$away_team_runs) * data$isHome

# get Opponent name
data$opponent <- NA
data$opponent[data$isHome == 1] <- data$away_team_name[data$isHome == 1]
data$opponent[data$isHome == -1] <- data$home_team_name[data$isHome == -1]

# drop canceled/tied games
data <- data[!is.na(data$score_diff) & data$score_diff != 0,]

# get wins
data$wins <- 0
data$wins[data$score_diff > 0] <- 1

# rows, cols
n_rows <- dim(data)[1]
n_cols <- dim(data)[2]

data <- with(data, data.frame(data,model.matrix(~opponent-1,data)))

# re order data
#data <- data[c(24, 23, 22, 21, 4:19)]

# write data
write.csv(dates, file=paste(target_team,"_dates.csv", sep=""), row.names = FALSE)
write.csv(data, file=paste(target_team,"_14_16.csv", sep=""), row.names = FALSE)
