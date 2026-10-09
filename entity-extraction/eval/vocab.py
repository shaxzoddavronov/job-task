"""Word lists for synthetic documents (plan section 11, Phase 2).

Invented organizations and people. Nothing here comes from a real document.
"""

ORGANIZATIONS = [
    "Oʻzbekiston Respublikasi Vazirlar Mahkamasi",
    "Oʻzbekiston Respublikasi Moliya vazirligi",
    "Oʻzbekiston Respublikasi Raqamli texnologiyalar vazirligi",
    "Oʻzbekiston Respublikasi Qurilish vazirligi",
    "Toshkent shahar hokimligi",
    "Samarqand viloyati hokimligi",
    "Oʻzbekiston Respublikasi Ekologiya vazirligi",
    "Oʻzbekiston Respublikasi Energetika vazirligi",
]

EXECUTORS = [
    "Moliya vazirligi",
    "Iqtisodiyot va moliya vazirligi",
    "Raqamli texnologiyalar vazirligi",
    "Qurilish vazirligi",
    "Toshkent shahar hokimligi",
    "Davlat soliq qoʻmitasi",
    "Statistika agentligi",
    "Adliya vazirligi",
    "viloyatlar hokimliklari",
    "Ekologiya vazirligi",
]

POSITIONS = [
    "Vazir",
    "Vazir oʻrinbosari",
    "Bosh direktor",
    "Hokim",
    "Hokim oʻrinbosari",
    "Boshqarma boshligʻi",
]

NAMES = [
    "A. Karimov", "B. Toshmatov", "D. Yusupova", "S. Raximov",
    "N. Abdullayev", "M. Ergashev", "G. Saidova", "T. Nazarov",
]

TITLES = [
    "Raqamli iqtisodiyotni rivojlantirish chora-tadbirlari toʻgʻrisida",
    "Byudjet intizomini mustahkamlash toʻgʻrisida",
    "Ekologik monitoringni takomillashtirish toʻgʻrisida",
    "Qurilish sohasida nazoratni kuchaytirish toʻgʻrisida",
    "Davlat xizmatlari sifatini oshirish toʻgʻrisida",
]

#: Task bodies. Each is a complete clause minus its deadline and executor,
#: which the generator appends.
ACTIONS = [
    "soha korxonalari faoliyati tahlil qilinsin",
    "hisobot Vazirlar Mahkamasiga taqdim etilsin",
    "zarur normativ-huquqiy hujjatlar loyihasi ishlab chiqilsin",
    "monitoring tizimi ishga tushirilsin",
    "mutaxassislar uchun oʻquv seminarlari tashkil etilsin",
    "byudjet mablagʻlari sarflanishi ustidan nazorat oʻrnatilsin",
    "elektron platformaga integratsiya taʼminlansin",
    "hududiy boʻlinmalar faoliyati muvofiqlashtirilsin",
    "aholi murojaatlari koʻrib chiqilsin",
    "statistik maʼlumotlar bazasi yangilansin",
    "xalqaro tajriba oʻrganilsin va takliflar kiritilsin",
    "ichki audit oʻtkazilsin",
]

PARENT_INTROS = [
    "quyidagilarni taʼminlasin",
    "quyidagi vazifalarni bajarsin",
    "quyidagi chora-tadbirlarni amalga oshirsin",
]

MONTHS = [
    "yanvar", "fevral", "mart", "aprel", "may", "iyun",
    "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr",
]
