# Finansal Danışman Chatbot Projesi

Bu proje, finansal danışmanlık alanında kullanıcıların sorularını yanıtlayan bir yapay zeka destekli chatbot geliştirmeyi amaçlamaktadır.  
Projede OpenAI GPT-3.5-turbo ve Ollama LLaMA 3.2 ve modelleri kullanılarak, kullanıcıların doğal dildeki finansal sorularına anlamlı ve güvenilir cevaplar verilmesi hedeflenmiştir.  
Veri seti olarak intent classification yapılmış finansal danışmanlık sohbet kayıtları kullanılmış ve bu veriler embedding yöntemiyle temsil edilip, arama ve yanıt üretme süreçlerinde kullanılmıştır.
Projede ayrıca chatbot performansını ölçmek için intent sınıflandırma doğruluğu ve sınıflandırma raporu hesaplanmıştır.  Son olarak da Streamlit üzerinden chatbot arayüzü ile chatbot deneyimi sunulmuştur.
Amaç, gerçek kullanıcı mesajlarından oluşan 1000 satırlık bir veri seti kullanarak RAG tabanlı chatbotlar geliştirmek ve bu modellerin performansını ölçmektir.


# Veri

Veri setinde her satırda şu alanlar yer alır:
- `intent`: Kullanıcının mesajındaki amacın kategorisi (örneğin: "tasarruf", "kredi", "sigorta" vb.)
- `kullanici_mesaji`: Kullanıcı tarafından yazılmış doğal dil mesajı
- `bot_cevabi`: Bu mesaja karşılık önerilen yanıt

Toplam 12 farklı intent kategorisi bulunmaktadır.
-   **selamlama**
-   **bilinmiyor**
-   **emeklilik**
-   **vedalasma**
-   **döviz**
-   **yatırım**
-   **borç_yönetimi**
-   **vergiler**
-   **sigorta**
-   **tasarruf**
-   **kredi**
-   **bütçe_planlama**

## Kullanılan modeller

- OpenAI GPT-3.5-turbo: LangChain ile OpenAI API'si üzerinden çalıştırılmıştır.
- Ollama LLaMA 3.2: Yerel ortamda `ollama` üzerinden çalıştırılmıştır.

## Uygulama Özellikleri

- RAG (Retrieval Augmented Generation) yapısı kullanılmıştır.
- Streamlit ile kullanıcı arayüzü hazırlanmıştır.
- Kullanıcı mesajına en yakın 5 belge vektör tabanlı olarak alınır.
- Yanıtlar GPT veya LLaMA modeli tarafından oluşturulur.

## Performans Ölçümü

- Veri %80 eğitim, %20 test olarak bölünmüştür.
- Modelin verdiği yanıtlar üzerinden intent tahmini yapılmıştır.
- Accuracy ve classification report (precision, recall, f1-score) hesaplanmıştır.

 Ollama LLaMA 3.2:
![Ollama Classification Report](https://i.imgur.com/z271adt.jpg)  

OpenAI GPT-3.5-turbo:
![OpenAI Classification Report](https://i.imgur.com/gQXziRq.jpg)

Kullanılan bu modeller ve Classification Reportlarına bakıldığında Ollama ile yapılan Chatbotun intent classification görevinde daha başarılı olduğu görülmüştür. İki modelin de düşük accuracy bulması ise veri sayısındaki düşüklük ve intent dağılımındaki oransızlık 
sebebiyle gerçekleşmiştir.

## Kullanıcı deneyimi ve Chatbot Ara Yüzü

Ollama ve OpenAI ile yapılan arayüz sonucunda OpenAI GPT-3.5-turbo daha iyi bir kullanıcı deneyimi sunmuştur. 
OpenAI GPT-3.5-turbo:
![OpenAI UI 1](https://i.imgur.com/qgogfRg.jpg) 
![OpenAI UI 2](https://i.imgur.com/oE8atF5.jpg)

Ollama ise OpenAI kadar olmasa da iyi sayılabilecek bir kullanıcı deneyimi sunmuştur.
Ollama LLaMA 3.2:
![Ollama UI 1](https://i.imgur.com/6wUBTL0.jpg) 
![Ollama UI 2](https://i.imgur.com/ofCWnB2.jpg)

# Sonuç
Bu projede hem Ollama LLaMA 3.2 hem de OpenAI GPT-3.5-turbo modelleri kullanılarak finansal danışmanlık alanında bir chatbot geliştirilmiş ve performansları karşılaştırılmıştır. Intent sınıflifikasyon görevi açısından Ollama modeli OpenAI’ye göre biraz daha iyi sonuçlar vermiştir. Ancak, kullanıcı deneyimi ve yanıtların akıcılığı bakımından OpenAI modeli daha başarılı bulunmuştur.

Her iki modelin de performansı, kullanılan veri setinin sınırlı büyüklüğü ve dengesiz intent dağılımından etkilenmiştir. Bu nedenle modellerin doğruluğunu artırmak için daha büyük ve dengeli veri setleri ile eğitilmeleri gerekmektedir. Ayrıca, RAG mimarisi ile metin tabanlı bilgi erişiminin chatbotların yanıt kalitesini artırdığı gözlemlenmiştir.

Genel olarak, bu çalışma hem yerel modellerin (Ollama) hem de bulut tabanlı güçlü LLM’lerin (OpenAI) finansal danışmanlık chatbotlarında kullanılabileceğini göstermiştir. Gelecekte daha geniş veri ve farklı model kombinasyonları ile performans ve kullanıcı deneyimi daha da geliştirilebilir.

