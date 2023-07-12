import tweepy
from allocator import portfolio_allocator

ema_filter = 0.05
symbols = ["SPY", "VXX", "NIO"]

# Authenticate to Twitter
auth = tweepy.OAuthHandler("API_KEY", "API_KEY_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")

# Create API object
api = tweepy.API(auth)

# Download stock data and compute optimal allocations
w, g, mu, sigma, VaR = portfolio_allocator(symbols, ema_filter)

# Create your tweet text
tweet_text = f"Optimal allocation for today: {w}"

# Tweet
api.update_status(status=tweet_text)
