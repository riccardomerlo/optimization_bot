git push heroku master

heroku ps:scale web=1

heroku logs -t