import cv2
import numpy as np

def pirinc_say(goruntu):
    # Görüntüyü gri seviyeye dönüştürür.
    gri_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    # Eşikleme işlemi ile sadece pirinçleri ayırır.
    eşik_deger = 127
    eşikli_goruntu = cv2.threshold(gri_goruntu, eşik_deger, 255, cv2.THRESH_BINARY)[1]

    # Morfolojik işlemler ile istenmeyen arka planları temizler.
    kernel = np.ones((3, 3), np.uint8)
    eşikli_goruntu = cv2.morphologyEx(eşikli_goruntu, cv2.MORPH_OPEN, kernel)

    # Büyütme ve küçültme işlemlerini kullanarak pirinç tanelerini birbirinden ayırır.
    büyütme_faktörü = 2
    eşikli_goruntu = cv2.resize(eşikli_goruntu, (0, 0), fx=büyütme_faktörü, fy=büyütme_faktörü)
    eşikli_goruntu = cv2.resize(eşikli_goruntu, (0, 0), fx=1 / büyütme_faktörü, fy=1 / büyütme_faktörü)

    # Sayma ve etiketleme fonksiyonları ile pirinç sayısını hesaplar.
    etiketler, sayi = cv2.connectedComponents(eşikli_goruntu)
    return sayi


# Görüntüyü yükler.
goruntu = cv2.imread("pirinc.jpeg")

# Pirinç sayısını hesaplar.
pirinc_sayisi = pirinc_say(goruntu)

# Pirinç sayısını yazdırır.
print("Pirinç sayısı:", pirinc_sayisi)
