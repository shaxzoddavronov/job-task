"""All user-facing bot texts (Uzbek Latin). No other bot module contains user-facing strings."""

EXAMPLE_QUESTIONS = [
    "ID-karta qancha muddatga beriladi?",
    "Haydovchilik guvohnomasini qanday olsam bo'ladi?",
    "Soliq qarzimni qanday tekshiraman?",
]

START = (
    "Assalomu alaykum! 👋\n\n"
    "Men davlat xizmatlari va huquqiy masalalar bo'yicha savollaringizga javob beruvchi botman: "
    "pasport va ID-karta, haydovchilik guvohnomasi, soliq, nafaqa, notarius, mulk va boshqalar.\n\n"
    "Savolingizni oddiy so'zlar bilan yozing (lotin yoki kirill alifbosida). "
    "Masalan, quyidagi savollardan birini bosing:"
)

HELP = (
    "Qanday so'rash kerak:\n"
    "• Savolni bitta xabarda, oddiy tilda yozing.\n"
    "• Lotin yoki kirill alifbosi — farqi yo'q.\n"
    "• Qisqartmalar ham tushuniladi: DXM, FHDYO, ERI, JSHSHIR, STIR, YaTT.\n"
    "• Agar aniq javob topilmasa, o'xshash savollarni taklif qilaman.\n\n"
    "Misollar:\n"
    "• ID-karta qancha muddatga beriladi?\n"
    "• prava olish uchun nima kerak\n"
    "• Солиқ қарзимни қандай текшираман?\n\n"
    "Buyruqlar: /start, /help, /about"
)

ABOUT = (
    "Bu bot sun'iy intellekt chat-modelisiz (LLM) va transformer modellarisiz ishlaydi.\n\n"
    "Usul: savol normallashtiriladi (kirill → lotin, apostroflar), qonun raqamlari va "
    "tashkilot nomlari ajratiladi, so'ng belgi n-gramm TF-IDF va statik so'z vektorlari "
    "(Model2Vec) yordamida eng yaqin javob qidiriladi. Natijalar qayta saralanadi va ishonch "
    "darajasi past bo'lsa, bot javob o'rniga takliflar beradi.\n\n"
    "{metrics}"
)
ABOUT_METRICS = (
    "Sinov natijalari:\n"
    "• Qo'lda yozilgan real savollarda to'g'ri javob (1-o'rin): {realistic_top1}\n"
    "• To'g'ri javob dastlabki 3 taklif ichida: {realistic_top3}\n"
    "• Mavzuga oid bo'lmagan savollarni rad etish: {ood_rejected}\n"
    "• O'rtacha javob vaqti: {latency}"
)
ABOUT_NO_METRICS = "Sinov natijalari hozircha mavjud emas."

NOT_FOUND = "Kechirasiz, savolingizni tushunmadim. Iltimos, boshqacha so'rab ko'ring."
UNKNOWN_LAW = (
    "Kechirasiz, {laws} bo'yicha ma'lumot bazamda yo'q. Noto'g'ri hujjat haqida javob "
    "bermaslik uchun javob qaytarmadim. Iltimos, hujjat raqamini tekshiring yoki savolni "
    "raqamsiz yozing."
)
SUGGEST = "Quyidagilardan birini nazarda tutdingizmi?"
API_DOWN = "Kechirasiz, xizmat hozir vaqtincha ishlamayapti. Birozdan so'ng qayta urinib ko'ring."
TOO_LONG = "Savol juda uzun. Iltimos, uni qisqaroq qilib yozing."
NOT_TEXT = "Iltimos, savolingizni matn ko'rinishida yozing."
ANSWER_MISSING = "Kechirasiz, bu javob topilmadi. Savolingizni qaytadan yozib ko'ring."

HELPFUL = "👍 Foydali"
NOT_HELPFUL = "👎 Foydali emas"
FEEDBACK_THANKS = "Rahmat! Fikringiz qabul qilindi."

COMMANDS = {
    "start": "Botni boshlash",
    "help": "Qanday so'rash kerak",
    "about": "Bot qanday ishlaydi",
}
