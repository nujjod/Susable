# Susable

## Inspiration

What inspired us is our environment. By providing cost effective solutions that help people save water and time that will contribute to better living conditions.

## What it does
Susable is able to automatically start/stop water pumps on the irrigation site based on the soil moisture content acquired from the moisture content, sensors will allow farmers to collect data about the weather and soil and we have built a Machine learning model to predict soil moisture in terms of soil temperature, air temperature and relative humidity. The model will allow for better water conservation while ensuring there isn’t over or underwatering of crops and save water, money and time.

## How we built it
We collected data about the weather and soil then export data to a real-time dashboard, we built a Machine learning model using Random Forest Regression to predict soil moisture in terms of soil temperature, air temperature and relative humidity and the sprinklers will work according to the predicted value, our system will send update alert to the farmer, also at the same time it will send command to an IoT application to start/stop the sprinkler. we got 76% r-squared value. We used AWS Simple notification service for the alert system. We used flask to deploy our model in our web application.
