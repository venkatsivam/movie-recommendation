from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import time
import pandas as pd

# set up webdriver path
web_driver_path = 'C:/webdriver/chromedriver-win64/chromedriver.exe'
service = Service(web_driver_path) # This line creates a Service object for the Chrome WebDriver
driver = webdriver.Chrome(service = service) # This line creates a WebDriver instance using the Service object
driver.get("https://www.imdb.com/search/title/?release_date=2020-01-01,2024-12-31")  # Open IMDb Movies page


# A function to click the "More" button until it's no longer available
def load_all_movies():
    while True:
        try:
            more_button = driver.find_element(By.CLASS_NAME, 'ipc-see-more__text')
            more_button.click()
            time.sleep(3)  # Wait for the new movies to load
        except Exception as e:
            print("No more movies to load.")
            break


load_all_movies()
movie_name_list = []
story_line_list = []
movie_details = driver.find_elements(By.CSS_SELECTOR, 'h3.ipc-title__text')
storyline_details = driver.find_elements(By.CSS_SELECTOR, '.ipc-html-content-inner-div')

for movie_name, storyline in zip(movie_details, storyline_details):
    movie_name_list.append(movie_name.text)
    story_line_list.append(storyline.text)

driver.quit()

# To save the data into dataframe
df = pd.DataFrame({"MovieName":movie_name_list, "StoryLine":story_line_list})

# To remove the numric chars in the movie name column
df['MovieName'] = df['MovieName'].str.replace(r'^\d+\.\s*', '', regex=True)

df.to_csv("movie_recommendations_2024_final.csv") # Saving the data intoc csv file for later use