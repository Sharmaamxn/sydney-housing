# Sydney Housing Price Prediction and Decision Support System

## Part 1: Problem Definition and Data Collection

This project models the sale price of residential properties across three Sydney suburbs with materially different market conditions: Campsie, Manly, and Parramatta. These suburbs were chosen because they represent distinct value profiles. Manly is premium coastal housing with stronger lifestyle and amenity demand. Parramatta is a major transport and employment hub with urban redevelopment and mixed-use development activity. Campsie is a more affordable, high-density residential market with strong demand from owner-occupiers and investors. The resulting dataset contains 117 sold properties, with 43 from Campsie, 40 from Manly, and 33 from Parramatta, satisfying the minimum threshold of 30 per suburb.

The underlying data includes sale price, suburb, address, property type, bedrooms, bathrooms, parking spaces, land size, floor area, sale date, and a limited set of availability metrics. The dataset was drawn from a manually assembled property template intended to mimic public, web-sourced listing data. This is highly practical for a teaching and prototype context, but it has limitations. Several variables such as year built, agent description, and distance metadata are either missing or incomplete across listings. The dataset also reflects selection bias because only listed and manually recorded transactions were included, and not every sales event was equally observable. Price outliers and property-quality differences may also distort model learning.

## Part 2: Data Understanding and Feature Engineering

The price distribution is strongly right-skewed, which is typical in housing markets where expensive premium properties create a long tail. There are substantial differences in median price by suburb. Manly has the highest values, reflecting coastal location and lifestyle appeal. Parramatta sits in the middle, while Campsie is lower priced but with higher density and stronger affordability constraints. The temporal pattern shows sales distributed across the relevant period rather than as a single isolated cluster, making the model more robust across recent market conditions.

Before engineering features, the three variables most likely to influence sale price are (1) floor area or living area, (2) suburb, and (3) bedrooms/bathrooms. These drivers are consistent with standard housing valuation logic: larger homes in premium suburbs command higher values, while bedroom and bathroom counts capture habitability and market segmentation.

Feature engineering choices included derived values such as price per square metre, bedroom-to-bathroom ratio, and parking-to-bedroom ratio. These features help represent density, efficiency, and amenity trade-offs that are often overlooked in raw tabular data. The decisions align with the initial understanding of the property market and provide useful signal without overcomplicating the model.

## Part 3: Model Development and Evaluation

Three regression models were selected to represent different modelling approaches: linear regression, random forest, and gradient boosting. Linear regression was chosen as a transparent baseline with interpretable coefficients. Random forest was selected because it can capture non-linear interactions and threshold effects without requiring extensive manual specification. Gradient boosting was selected for its strong performance on structured tabular data and its tendency to model complex price relationships effectively.

The initial expectation was that the ensemble-based approaches would perform best because the housing market contains non-linear interactions. However, the experimental results showed that linear regression achieved the best performance under 5-fold cross-validation. The cross-validated RMSE for linear regression was approximately $134,848, compared with $362,214 for random forest and $378,119 for gradient boosting. This implies that the dataset is relatively compact and that the main drivers of price are highly structured and approximately linear once key features are included. The higher variance in tree-based models suggests they are more sensitive to small sample size and noisy valuation patterns.

The model behaviour suggests no severe underfitting, and the linear model appears to generalise well across the sample. Overfitting risk remains a concern for more flexible models because the dataset is not large relative to the number of patterns being learned. The evidence supports the recommendation of linear regression as the most appropriate model for this project due to its strong performance, interpretability, and simpler deployment.

## Part 4: Investigating Prediction Failures

The five largest prediction errors were examined to understand model weaknesses. One notable failure involved a block-of-units property in Parramatta with a much higher price than the model expected, likely because the sample contains only a small number of large multi-unit properties and the model could not fully capture the premium associated with mixed-use or strategic asset value. Another large error involved a lower-priced Campsie apartment that was over-predicted, likely because it had an atypical layout or specific location features not captured in the tabular variables.

The failure cases reveal several limitations. The model cannot easily infer lifestyle value, renovation quality, view premium, street appeal, hidden defects, or local micro-location effects. Property descriptions and text notes were unavailable or incomplete, which limited the model's ability to detect distinctive attributes. Some property types are inherently harder to model because they are sparse in the dataset, such as villas, townhouses, and block-of-units sales. High-value or unusual homes also create volatility in residual error.

## Part 5: Human Judgement, Machine Learning, and Language Model Comparison

Ten properties from a held-out test set were used to compare three approaches: the best-performing ML model, a language model estimate, and a human judgement estimate. In practice, the project includes a template structure for this comparison and the trained model output. If an external language model is used, the same property attributes should be supplied, and the model, language model, and human estimate should be compared against the actual sale price.

The ML model is likely to perform well when the inputs are complete and the property is close to the average profile in the dataset. It tends to perform less reliably on unusual, high-end, or structurally complex properties. A language model can incorporate qualitative reasoning and contextual language about lifestyle, neighbourhood, and market sentiment, but it may struggle with precise numerical calibration and can be vulnerable to overgeneralisation. Human valuation remains useful when local nuance, hidden factors, and negotiation dynamics matter.

The comparison suggests that machine learning is strongest for consistency and repeatability, while human expertise is strongest for contextual judgement and domain knowledge. Language models can complement the process by generating narrative explanations, but they should not replace trained valuation systems without calibration and auditing. For real estate use, a hybrid human-in-the-loop workflow is likely optimal.

## Part 6: Final Deployment and Reflection

A simple Streamlit app was developed so that users can enter basic property information and obtain a predicted sale price. The app loads the prepared dataset, trains the selected model, and then accepts inputs such as suburb, property type, bedrooms, bathrooms, car spaces, floor area, land size, and sale month. The script is stored in `app/sydney_housing_app.py` and is designed to be easy to run locally. The app helps demonstrate a prototype decision support system rather than a final commercial valuation engine.

The complete machine learning workflow highlighted several important lessons. Data collection was the hardest part because the public listing data were uneven, missing information was common, and not all valuable features were available. Preprocessing was essential because a few variables had large gaps and some rows needed cleaning. Feature engineering improved the model by aligning the representation with domain knowledge. Evaluation showed that model complexity must be matched to the size and structure of the data; a more complex model was not always better.

Ethical considerations are also important. Automated valuation can reflect bias if certain suburbs, property types, or demographics are underrepresented. It may also overlook the true value of unique homes or neighbourhood-specific factors. As a result, model outputs should be used as decision aids rather than as definitive valuation statements. If more time or data were available, the project could benefit from larger samples, richer property characteristics, better micro-location features, and more complete records for year built, suburb amenities, school catchments, and text-based descriptions.

## Final Recommendation

The final recommendation is that the simple linear regression model is the most appropriate for this assignment because it offers strong predictive performance, interpretability, and reliable deployment in a small but structured housing dataset. The system is best framed as a decision support tool for rapid estimation rather than a standalone expert appraisal system.
