// --- 1. SÖZLÜK VE MODEL DEĞİŞKENLERİ ---
let eTicaretSozluk = []; // Boş başlıyor, JSON'dan dolacak
let globalModel = null;
// Sayfa açıldığında sözlüğü dışarıdan çeken fonksiyon
async function sozleriYukle() {
    try {
        const res = await fetch('sozluk.json');
        eTicaretSozluk = await res.json();
        console.log("Sözlük başarıyla yüklendi. Kelime sayısı:", eTicaretSozluk.length);
    } catch (err) {
        console.error("Sözlük yüklenirken hata oluştu:", err);
    }
}
// Uygulama başlar başlamaz sözlüğü çekiyoruz
sozleriYukle();
// Metni 0 ve 1'lerden oluşan vektöre (Tensör girdisine) çeviren fonksiyon
function metniVektoreCevir(metin) {
    if (eTicaretSozluk.length === 0) return [];
    
    // Görseldeki kelimeler Türkçe karaktersiz olduğu için girdiyi de İngilizce karakter yapıyoruz
    let temizMetin = metin.toLowerCase()
        .replace(/ç/g, 'c').replace(/ğ/g, 'g')
        .replace(/ı/g, 'i').replace(/ö/g, 'o')
        .replace(/ş/g, 's').replace(/ü/g, 'u');
        
    return eTicaretSozluk.map(kelime => temizMetin.includes(kelime) ? 1 : 0);
}

// Metnin içindeki yakalanan kelimeleri ekrana basmak için filtreleyen fonksiyon
function yakalananKelimeleriBul(metin) {
    if (eTicaretSozluk.length === 0) return [];
    
    let temizMetin = metin.toLowerCase()
        .replace(/ç/g, 'c').replace(/ğ/g, 'g')
        .replace(/ı/g, 'i').replace(/ö/g, 'o')
        .replace(/ş/g, 's').replace(/ü/g, 'u');
        
    return eTicaretSozluk.filter(kelime => temizMetin.includes(kelime));
}

// --- 2. TENSORFLOW.JS MODEL EĞİTİMİ ---
async function modeliEgit() {
    if (eTicaretSozluk.length === 0) {
        await sozleriYukle();
    }

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [eTicaretSozluk.length] }));
    model.add(tf.layers.dense({ units: 4, activation: 'sigmoid' })); // [Kargo, Ürün, Pozitif, Negatif]

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });

    // CÜMLELERİ BIRAKTIK! Kelime gruplarını doğrudan matrise çeviriyoruz.
    const egitimGirdileri = tf.tensor2d([
        // 1. Grup: Saf Kargo Kelimeleri Örnekleri
        metniVektoreCevir("kargo yavas teslimat rezalet kurye berbat ulasmadi gec"),
        metniVektoreCevir("kargo hizli teslimat guzel zamaninda ulasti paketleme harika"),

        // 2. Grup: Saf Ürün Kelimeleri Örnekleri
        metniVektoreCevir("urun kaliteli kumas beden guzel kalitesi muhtesem"),
        metniVektoreCevir("urun defolu dikisleri berbat yirtik bozuk dandik plastik kalitesiz"),

        // 3. Grup: Saf Olumlu/Negatif Duygu Tetikleyicileri
        metniVektoreCevir("harika mukemmel tesekkurler begendim begendi iyiki tavsiye"),
        metniVektoreCevir("cop rezalet iade kotu yazik onermiyorum hic degil yok kandirmislar resmen fotoda") 
    ]);

    // Çıktı Etiketleri: [Kargo, Ürün, Pozitif, Negatif]
    const egitimEtiketleri = tf.tensor2d([
        [1, 0, 0, 1], // Kargo ağırlıklı kelimeler (Genel negatif)
        [1, 0, 1, 0], // Kargo ağırlıklı kelimeler (Genel pozitif)
        
        [0, 1, 1, 0], // Ürün ağırlıklı kelimeler (Genel pozitif)
        [0, 1, 0, 1], // Ürün ağırlıklı kelimeler (Genel negatif)
        
        [0, 0, 1, 0], // Sadece saf pozitif kelimeler tetiği
        [0, 0, 0, 1]  // Sadece saf negatif kelimeler tetiği (kandırmışlar, fotoda, yok, değil burada ağırlık kazandı!)
    ]);

    // Kelime bağlarını koparmamak için epoch sayısını 150 yapıyoruz
    await model.fit(egitimGirdileri, egitimEtiketleri, { epochs: 150 });
    return model;
}

// --- 3. ARAYÜZ ETKİLEŞİMLERİ (DOM MANİPÜLASYONU) ---
document.getElementById('train-btn').addEventListener('click', async () => {
    const modal = document.getElementById('training-modal');
    const modelStatus = document.getElementById('model-status');
    const analyzeBtn = document.getElementById('analyze-btn');

    // 1 Dakikalık eğitim animasyonunu başlat (Modalı aç)
    modal.classList.remove('hidden');

    try {
        // TensorFlow modelini arka planda eğit
        globalModel = await modeliEgit();
        
       
        setTimeout(() => {
            modal.classList.add('hidden');
            modelStatus.innerHTML = "Model durumu: Eğitim Tamamlandı ve Hazır ✅";
            modelStatus.style.color = "#2ecc71";
            analyzeBtn.disabled = false; // Analiz butonunu aktif et
        }, 1500);

    } catch (error) {
        console.error("Model eğitilirken hata oluştu:", error);
        modal.classList.add('hidden');
        modelStatus.innerHTML = "Model durumu: Hata Oluştu ❌";
    }
});

document.getElementById('analyze-btn').addEventListener('click', async () => {
    const userInput = document.getElementById('review-input').value;
    const categoryOutput = document.getElementById('category-output');
    const sentimentOutput = document.getElementById('sentiment-output');
    const keywordsContainer = document.getElementById('keywords-container');

    if (!userInput.trim() || !globalModel) return;

    // 1. HER ANALİZDEN ÖNCE ARAYÜZÜ VE SINIFLARI TAMAMEN SIFIRLA
    categoryOutput.className = "badge badge-empty";
    categoryOutput.innerText = "Hesaplanıyor...";
    sentimentOutput.className = "badge badge-empty";
    sentimentOutput.innerText = "Hesaplanıyor...";
    keywordsContainer.innerHTML = "";

    // 2. TF.TIDY İLE HAFIZAYI VE TENSÖRLERİ KORUMA ALTINA ALIYORUZ
    // Bu sayede arka arkaya basıldığında eski tahminler yeni tahmini ASLA kirletemez.
    const sonuclar = tf.tidy(() => {
        const vektor = metniVektoreCevir(userInput);
        const girdiTensor = tf.tensor2d([vektor]);
        const tahmin = globalModel.predict(girdiTensor);
        
        // Veriyi senkronize ve temiz bir şekilde dışarı fırlatıyoruz
        return tahmin.dataSync(); 
    });

    // 3. 1. Katman Sonucunu Ekrana Bas (Kargo mu Ürün mü?)
    categoryOutput.className = "badge"; // Temizle
    if (sonuclar[0] >= sonuclar[1]) {
        categoryOutput.innerText = "Kargo / Teslimat 📦";
        categoryOutput.classList.add('badge-kargo');
    } else {
        categoryOutput.innerText = "Ürün Kalitesi 💎";
        categoryOutput.classList.add('badge-urun');
    }

    // 4. 2. Katman Sonucunu Ekrana Bas (Pozitif mi Negatif mi?)
    sentimentOutput.className = "badge"; // Temizle
    if (sonuclar[2] >= sonuclar[3]) {
        sentimentOutput.innerText = "Olumlu / Memnun 🌱";
        sentimentOutput.classList.add('badge-pozitif');
    } else {
        sentimentOutput.innerText = "Olumsuz / Şikayet 🚨";
        sentimentOutput.classList.add('badge-negatif');
    }

    // 5. Metinde Yakalanan Anahtar Kelimeleri Ekrana Bas
    const sampleText = userInput; // Referans temizliği
    const yakalananlar = yakalananKelimeleriBul(sampleText);

    if (yakalananlar.length === 0) {
        keywordsContainer.innerHTML = '<span class="empty-tag">Eşleşen anahtar kelime bulunamadı.</span>';
    } else {
        yakalananlar.forEach(kelime => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.innerText = kelime;
            keywordsContainer.appendChild(tag);
        });
    }
});