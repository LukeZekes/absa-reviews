from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from math import ceil
import regex

HEADERS = ({'User-Agent': 
	'''Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36''', 
	'Accept-Language': 'en-US, en;q=0.5'})

# Give the URL for a product page on Amazon, and this code will automatically navigate to the reviews and read them
def get_reviews_for_product(url):
	reviews_tags = []
	reviews = []
	driver = webdriver.Chrome()
	driver.get(url)

	# Find link to reviews page
	view_reviews_link = driver.find_element(By.CSS_SELECTOR, "#reviews-medley-footer > .a-spacing-medium > a").get_attribute("href")
	# Navigate to review page
	driver.get(view_reviews_link)
	# Find total number of product reviews
	total_num_reviews_text = driver.find_element(By.CSS_SELECTOR, "div#filter-info-section").text
	total_num_reviews_text_match = regex.search(r"\d+ with reviews", total_num_reviews_text)
	total_num_reviews_text_sub = total_num_reviews_text_match.group()
	num_size = 0
	for c in total_num_reviews_text_sub:
		if c.isdigit():
			num_size+=1
					
	num_available_reviews = min(100, int(total_num_reviews_text_sub[:num_size]))  # Right now it looks like Amazon will only show you up to 100, in 10 pages of 10 reviews each
	num_pages = ceil(num_available_reviews/10) # 10 reviews per page
	# Override
	num_pages = 2
	pagination_bar = driver.find_element(By.ID, "cm_cr-pagination_bar")
	next_page_button = pagination_bar.find_element(By.CSS_SELECTOR, "ul > li.a-last")

	for i in range(1, num_pages+1):
		# Locate reviews
		review_container = driver.find_element(By.ID, "cm_cr-review_list")
		review_web_elements = review_container.find_elements(By.CSS_SELECTOR, "div.review > div > div")

		# Extract relevant information from reviews
		for r in review_web_elements:
			# Filter out non-English reviews
			is_english = False
			try:
				r.find_element(By.CLASS_NAME, "cr-translate-this-review-section")
			except NoSuchElementException:
				is_english = True
			if is_english:
				reviews.append(review_as_dict(r))

		# Move to next page (if there is one)
		if i < num_pages:
			driver.get(view_reviews_link + "&pageNumber=" + str(i + 1))

	driver.quit()
	return reviews

def review_as_dict(tag):
	# Extracting review text
	id = tag.get_attribute("id")
	selector_string = "div#" + id + " > *" # Used to select only direct children
	tag_children = tag.find_elements(By.CSS_SELECTOR, selector_string)


	review_text = tag_children[4].text

	# Extracting rating
	rating_text = tag_children[1].find_element(By.CSS_SELECTOR, "i > span").get_attribute("innerHTML")
	rating = float(rating_text[:3])

	# Extracting date
	date_string = tag_children[2].text

	month_match = regex.search(r"January|February|March|April|May|June|July|August|September|October|November|December", date_string)
	month = month_match.group()
	month_end_index = month_match.end()

	day_index = month_end_index + 1
	day = date_string[day_index:day_index+2]
	if day[1] == ',':
		# Day is a 1 digit number
		day = day[0]
		year_index = month_end_index + 4
	else:
		year_index = month_end_index + 5
      
	year = date_string[year_index: year_index+4]

	date = {
		'month':month,
		'day':day,
		'year': year
	}



	r = {
		'text': review_text,
		'rating': rating,
		'date': date
	}

	return r