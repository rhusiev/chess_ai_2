mkdir data
cd data
mkdir 20.11
curl https://database.lichess.org/standard/lichess_db_standard_rated_2020-11.pgn.zst --output lichess_2020-11.pgn.zst
pzstd -d lichess_2020-11.pgn.zst
cd ..
