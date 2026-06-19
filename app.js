// --- 1. SÖZLÜK VE MODEL DEĞİŞKENLERİ ---
let eTicaretSozluk = []; // Boş başlıyor, JSON'dan dolacak
let globalModel = null;

function metniNormalizeEt(metin) {
    if (typeof metin !== 'string') {
        return '';
    }

    return metin
        .toLowerCase()
        .normalize('NFKD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/ç/g, 'c').replace(/ğ/g, 'g')
        .replace(/ı/g, 'i').replace(/ö/g, 'o')
        .replace(/ş/g, 's').replace(/ü/g, 'u')
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}
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
    
    const temizMetin = metniNormalizeEt(metin);

    return eTicaretSozluk.map(kelime => temizMetin.includes(kelime) ? 1 : 0);
}

// Metnin içindeki yakalanan kelimeleri ekrana basmak için filtreleyen fonksiyon
function yakalananKelimeleriBul(metin) {
    if (eTicaretSozluk.length === 0) return [];
    
    const temizMetin = metniNormalizeEt(metin);

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
        // 1. Kargo / Teslimat + Olumlu Kelimeler
        metniVektoreCevir(
            "basarili cabuk ertesi geldi gonderdiler gonderildi hemen hizli hizliydi " +
            "kargo kargoda kargolama kargoya siparis surede teslim teslimat trendyol " +
            "ulasti zaman zamaninda satici gayet iyi iyiydi memnun memnunum tesekkur tesekkurler"
        ),

        // 2. Kargo / Teslimat + Olumsuz Kelimeler
        metniVektoreCevir(
            "acilmadi acilmiyor bekledigim bekledigimden beklemeyin berbat bozuk bozuldu " +
            "calismadi calismiyor eksik gec gelmedi gelmiyor gonderdim hatali iade " +
            "kalmadim malesef maalesef problem rezalet rezil sikinti sorun sorunsuz " +
            "yok yoktu gecikme satici aldanmayin almayin vermeyin yanlis"
        ),

        // 3. Ürün Kalitesi + Olumlu Kelimeler
        metniVektoreCevir(
            "acik acildi alinabilir alisveris bayildi bayildim " +
            "begendi begendik begendim begenerek cihaz calisiyor " +
            "deger duzgun efsane fiyat fiyata fiyati fiyatina gercekten gorundugu " +
            "gorunuyor guzel guzeldi harika hediye hos hosuma ince kalite kalitede " +
            "kaliteli kalitesi kullandim kullanilabilir kullanim kullanisli kullaniyorum " +
            "kumas kumasi kutu kutusu lazim magaza makina makine makinesi malzeme " +
            "malzemesi muhtesem mukemmel muthis normal oldu olmus oneririm ozenli " +
            "paketi paketleme paketlemesi performans performansi puan rahat saglam " +
            "sahane sarj sarji satin ses sevdi sevdim suan super tavsiye tekrar " +
            "telefon ucuz urun urunde urunden urunler urunu urunum urunun uygun " +
            "uzun var vardi verdim veriyor yeni yer yeterli yildiz yine yorum yorumlari yuksek zaten"
        ),

        // 4. Ürün Kalitesi + Olumsuz Kelimeler
        metniVektoreCevir(
            "aldanmayin almayin begenmedim berbat bozuk bozuldu cizik cop cope " +
            "dandik defolu degil degmez dikisleri dikkat dusuk igrenc kalitesiz " +
            "kirik kirildi kokuyor kotu kotuydu kucuk kullanissiz kullanmadim " +
            "malesef maalesef olmadi olmuyor onermiyorum ozensiz pahali param " +
            "paraniza parasina pisman pismanim plastik problem rezalet rezil " +
            "sikinti sorun yirtik yirtildi yok yoktu zor zorunda " +
            "eksik hatalı hatali hasar hasarli bozulmus koptu teslimat " +
            "kalitesiz ucuz parasizilik ucuza bekledigim beklenigim " +
            "acikti acildi coruk kirmis kirtildi seklinde sekilsiz ince ipek nici " +
            "kullanilmiyor kullanilmaz yapilmamis yapilmiyor yapilmas " +
            "zarar zedeli zedelemis ziyan ziyanda oynatilmis oynatilmiyor " +
            "sinifi sinifinda derecesi derecesinde seviyesi seviyesinde " +
            "kapasite kapasitesi ozellik ozellikle tanitim tanitimda tanitiminda " +
            "tekrar tekrarli tekrarlanma tekrarlandi " +
            "teslimata teslimatte teslimatta teslimatinde teslimatinda " +
            "ucuzu ucuzunuz ucuzum malzeme malzemes malzemeleri malzemenin " +
            "seklini seklinde seklindeki sekline seklinin seklinde " +
            "aciklamasi aciklamada aciklamadaki aciklamalar aciklama " +
            "baskasina baskasinin baskasina baskalarina baskalarin " +
            "gelmedi gelmedigi gelmeyen gelmiyor gelmiş gelişi gelişinde " +
            "calismiyor calismiyor calismadi calismayan calismayacak " +
            "sahte sahtedir sahtesi sahtecilik sahte sakte saktedir " +
            "hile hileli hilenin hileyedi hileci hildeci hildeciler " +
            "reklam reklamda reklamdan reklamlari reklamli reklamın " +
            "goturu goturuu goturmedi goturmemiş goturmeyiz goturemiyor " +
            "donen donemi doneminde doneminin doneminin donemin doneminize " +
            "acilabilir acilabilirse acilmadi acilmadiği acilmiyor acilmayacak " +
            "satilan satildi satilmadi satilmiş satilmiyor satilacak satilacagi"
        )
    ]);

    // Çıktı Etiketleri: [Kargo, Ürün, Pozitif, Negatif]
    const egitimEtiketleri = tf.tensor2d([
        [1, 0, 1, 0], // 1. Grup: Kargo + Pozitif
        [1, 0, 0, 1], // 2. Grup: Kargo + Negatif
        [0, 1, 1, 0], // 3. Grup: Ürün + Pozitif
        [0, 1, 0, 1]  // 4. Grup: Ürün + Negatif
    ]);

    // Kelime bağlarını koparmamak için epoch sayısını 300 yapıyoruz (negatif kelimeleri daha ağır öğrenmesi için)
    await model.fit(egitimGirdileri, egitimEtiketleri, { epochs: 300 });
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

    if (!userInput.trim() || !globalModel || eTicaretSozluk.length === 0) {
        keywordsContainer.innerHTML = '<span class="empty-tag">Model veya sözlük hazır değil.</span>';
        return;
    }

    // 1. HER ANALİZDEN ÖNCE ARAYÜZÜ VE SINIFLARI TAMAMEN SIFIRLA
    categoryOutput.className = "badge badge-empty";
    categoryOutput.innerText = "Hesaplanıyor...";
    sentimentOutput.className = "badge badge-empty";
    sentimentOutput.innerText = "Hesaplanıyor...";
    keywordsContainer.innerHTML = "";

    // 2. TF.TIDY İLE HAFIZAYI VE TENSÖRLERİ KORUMA ALTINA ALIYORUZ
    // Bu sayede arka arkaya basıldığında eski tahminler yeni tahmini ASLA kirletemez.
    let sonuclar;
    try {
        sonuclar = tf.tidy(() => {
            const vektor = metniVektoreCevir(userInput);
            if (vektor.length !== eTicaretSozluk.length) {
                throw new Error('Girdi vektörü sözlük boyutuyla eşleşmiyor.');
            }

            const girdiTensor = tf.tensor2d([vektor]);
            const tahmin = globalModel.predict(girdiTensor);

            // Veriyi senkronize ve temiz bir şekilde dışarı fırlatıyoruz
            return tahmin.dataSync();
        });
    } catch (error) {
        console.error('Analiz sırasında hata oluştu:', error);
        categoryOutput.className = 'badge badge-empty';
        categoryOutput.innerText = 'Analiz yapılamadı';
        sentimentOutput.className = 'badge badge-empty';
        sentimentOutput.innerText = 'Analiz yapılamadı';
        keywordsContainer.innerHTML = '<span class="empty-tag">Bu yorum işlenemedi.</span>';
        return;
    }

    const threshold = 0.55; // %55 barajı (model dengesizliğine karşı daha katı)

    const yakalananKelimeler = yakalananKelimeleriBul(userInput);
    if (yakalananKelimeler.length > 0) {
        keywordsContainer.innerHTML = yakalananKelimeler
            .map(kelime => `<span class="tag">${kelime}</span>`)
            .join('');
    } else {
        keywordsContainer.innerHTML = '<span class="empty-tag">Eşleşen anahtar kelime bulunamadı.</span>';
    }

    const kargoSkor = sonuclar[0];
    const urunSkor = sonuclar[1];
    const pozitifSkor = sonuclar[2];
    const negatifSkor = sonuclar[3];

    // 1. Katman Kararı: Konu / Kategori
    categoryOutput.className = "badge"; // Reset class
    
    if (kargoSkor >= threshold && kargoSkor > urunSkor) {
        categoryOutput.innerText = `Kargo / Teslimat 📦 (%${Math.round(kargoSkor * 100)})`;
        categoryOutput.classList.add('badge-kargo');
    } else if (urunSkor >= threshold && urunSkor > kargoSkor) {
        categoryOutput.innerText = `Ürün Kalitesi 💎 (%${Math.round(urunSkor * 100)})`;
        categoryOutput.classList.add('badge-urun');
    } else {
        // İki ihtimal de barajın altındaysa veya tamamen eşitse
        categoryOutput.innerText = "Genel / Belirsiz 🔍";
        categoryOutput.classList.add('badge-empty');
    }

    // 2. Katman Kararı: Duygu Durumu
    sentimentOutput.className = "badge"; // Reset class
    
    if (pozitifSkor >= threshold && pozitifSkor > negatifSkor) {
        sentimentOutput.innerText = `Olumlu / Memnun 🌱 (%${Math.round(pozitifSkor * 100)})`;
        sentimentOutput.classList.add('badge-pozitif');
    } else if (negatifSkor >= threshold && negatifSkor > pozitifSkor) {
        sentimentOutput.innerText = `Olumsuz / Şikayet 🚨 (%${Math.round(negatifSkor * 100)})`;
        sentimentOutput.classList.add('badge-negatif');
    } else {
        // Kullanıcı nötr bir şey yazdıysa veya model kararsız kaldıysa
        sentimentOutput.innerText = "Nötr / Belirsiz 😐";
        sentimentOutput.classList.add('badge-empty');
    }
});