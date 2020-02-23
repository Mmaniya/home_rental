import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getJSONLinksFromContentReponse(Response):

        docHtmlContent = Response.content #Getting the body content of the httpResponse
        html_soup = BeautifulSoup(docHtmlContent, 'html.parser') # Parsing the Response Body
        section = html_soup.find("section", {"class": "tabsContent block-white dontSwitch"}) #Searching for section containing list of announcement
        try :
            ul = section.find("ul") # Getting List of the announcements
            li = ul.findAll("li") # Getting announcements Elements from the previous List
        except:
            return []
        announcementsList = [] # List of announcements
        for item in li:
            '''
            Parsing each announcement and
            Making a JSON which contains the details.
            Cleaning Data for the URL of the announcement,
            And the Category of the annoucement
            and preparing it for a Future use
            '''
            announcementsList.append({ 'PostTitle' : item.find('a')['title'],'link' : "https:"+(item.find('a')['href'])})
        return announcementsList



def proceedAnalysis(theAnnouncementsJSONList,isProfessionalAnnouncement):

    titleList= []
    postLinkList= []
    rentAmountList= []
    buildingPostCodeList= []
    posthasPhoneContactList= []
    posthasMailContactList= []
    postNumberOfPhototList= []
    postDescriptionList= []
    isBuildingfurnishedList= []
    buildingAreaList= []
    buildingRommsNumberList= []
    isProfessionalAnnouncementList = []

    if(isProfessionalAnnouncement) :
        print("Parsing scraped HTML Responses for professional announcements")
    else :
        print("Parsing scraped HTML Responses for inidvidual announcements")

    collected = 0
    collectedSuccess = 0
    problemPrice= 0
    for articleJSON in theAnnouncementsJSONList:
        title = articleJSON["PostTitle"]
        postLink = articleJSON["link"]
        pageAnnouncementResponse = requests.get(postLink) # Opening the url of the announcement
        if(pageAnnouncementResponse.ok) : #Announcement Response is well
            docHtmlContent = pageAnnouncementResponse.content #Getting the body content of the httpResponse
            #Solving Problems with encoding
            try :
                DOM = html.fromstring(docHtmlContent.decode('utf8'))  # DOM is the html document oriented object representation
            except UnicodeDecodeError :
                DOM = html.fromstring(docHtmlContent.decode('latin-1'))  # DOM is the html document oriented object representation

            MaybeInsertedIntoDataFrame = True #boolean which will control saving of the data (Yes/No if there's needed fields)
            #Getting tagged (id,class or itemprop) fields
            #Getting the Rent Amount
            try :
                rentAmount = int("".join(re.findall('\d+', (DOM.xpath('//h2[@itemprop="price"]/span/text()')[1]))))
            except :
                MaybeInsertedIntoDataFrame = False

            #Getting Post Code of the building
            try :
                buildingPostCode = int(re.findall('\d+',DOM.xpath('//span[@itemprop="address"]/text()')[0])[0])
            except IndexError :
                buildingPostCode = "NaN"

            #Getting number of photos of the announcement (There is a problem when the post has no photo si we catch this case)
            try:
                postNumberOfPhoto = ((re.findall('\d+',DOM.xpath('//p[@class="item_photo"]/text()')[0]))[0])
            except:
                #When the Post hasn't any photo
                if(len(DOM.xpath('//div[@class="item_image empty"]'))):
                    postNumberOfPhoto = 0
                else :
                    postNumberOfPhoto = 1

            #Getting the number of the available contacts in the announcement
            postNumberOfContact = len(DOM.xpath('//div[@class="box-grey-light mbs align-center"]/div')[0].getchildren())
            posthasPhoneContact = len(DOM.xpath('//button[@class="button-orange large phoneNumber trackable"]'))
            posthasMailContact = 0
            if(postNumberOfContact - posthasPhoneContact > 0):
                posthasMailContact = 1

            #Getting the description of the announcement
            try :
                postDescription = ' '.join(DOM.xpath('//p[@itemprop="description"]/text()'))
            except :
                postDescription  = "NaN"




            # Getting non Tagged fields by Text Content
            isBuildingfurnished = "NaN" #default value "non meublé"
            buildingArea = "NaN" #default Value in m²
            buildingRommsNumber = "NaN" #default rooms number equals 1
            othersNonTaggedFields = DOM.xpath('//section[@class="properties lineNegative"]/div[@class="line"]/h2[@class="clearfix"]/span/text()')[:8]

            #Checking if other informations exist
            FoundOtherFields = True
            for elt in ["Meublé / Non meublé","Pièces","Surface"]:
                FoundOtherFields = FoundOtherFields and (elt in othersNonTaggedFields)

            for cpt in range(0,len(othersNonTaggedFields)):
                key = (othersNonTaggedFields[cpt].strip())
                if key in ["Meublé / Non meublé","Pièces","Surface"]: # Ces Champs nous concernent pour la suite
                    value = othersNonTaggedFields[cpt+1].strip()
                    if((key == "Meublé / Non meublé")):
                        isBuildingfurnished = 0
                        if (value == "Meublé"):
                            isBuildingfurnished = 1
                    if(key == "Pièces"):
                        buildingRommsNumber = int(value)
                    if(key == "Surface"):
                        buildingArea = re.findall('\d+',value)[0]


            ### Cleaning Data ###
            #Treatment of aberrant values, we won't insert illogic values in our dataFrame
            squareMeterRent = (float(rentAmount)/float(buildingArea))

            #Inserting only completely  Rows with a present Price field
            if(MaybeInsertedIntoDataFrame):
                titleList.append(title)
                postLinkList.append(postLink)
                rentAmountList.append(rentAmount)
                buildingPostCodeList.append(buildingPostCode)
                postNumberOfPhototList.append(postNumberOfPhoto)
                posthasPhoneContactList.append(posthasPhoneContact)
                posthasMailContactList.append(posthasMailContact)
                postDescriptionList.append(postDescription)
                isBuildingfurnishedList.append(isBuildingfurnished)
                buildingAreaList.append(buildingArea)
                buildingRommsNumberList.append(buildingRommsNumber)
                isProfessionalAnnouncementList.append(isProfessionalAnnouncement) #Extracting from PRO agencies so there is no fraudes
    print("Contructing Data Frame")
    df_Announcements = pd.DataFrame(list(zip(buildingPostCodeList,buildingRommsNumberList,postNumberOfPhototList,posthasMailContactList,posthasPhoneContactList,postDescriptionList,isBuildingfurnishedList,buildingAreaList,isProfessionalAnnouncementList,rentAmountList)),\
    columns = ['Zip Code','Rooms Number','NbPhotos','has_mail','has_phone','description','is furnished','Area','Rent from Agency','Rent(€)'])
    return df_Announcements


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)

train.head()


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train.SalePrice.describe()

#print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

target = np.log(train.SalePrice)
#print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

corr = numeric_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])

train.OverallQual.unique()

quality_pivot = train.pivot_table(index='OverallQual',values='SalePrice', aggfunc=np.median)

quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
#plt.show()

plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
#plt.show()

plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
#plt.show()

train = train[train['GarageArea'] < 1200]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
#plt.show()

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()

#print ("Original: \n")
#print (train.Street.value_counts(), "\n")

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

#print ('Encoded: \n')
#print (train.enc_street.value_counts())

condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

sum(data.isnull().sum() != 0)

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)



from sklearn import linear_model
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
print(X_test.head(1))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()



for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()



submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()

print(feats)
predictions = model.predict(feats)

final_predictions = np.exp(predictions)


print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()

submission.to_csv('submission1.csv', index=False)
