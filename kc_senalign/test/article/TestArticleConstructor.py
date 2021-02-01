import unittest

from src.article.KhmerArticle import KhmerArticle
from src.article.VietnameseArticle import VietnameseArticle


class ArticleConstructorTestCase(unittest.TestCase):
    def test_km_constructor(self):
        article = KhmerArticle(
            "នរដ្ឋាភិបាលវៀតណាមបន្តពង្រឹងវិធានការបង្ការនិងប្រយុទ្ធប្រឆាំងនឹងជំងឺរាតត្បាតកូវីដ ១៩\n ៅក្នុងកិច្ចប្រជុំគណៈកម្មាធិការជាតិគ្រប់គ្រងការងារបង្ការនិងប្រយុទ្ធប្រឆាំងជំងឺរាតត្បាត កូវីដ ១៩ ដែលបានធ្វើឡើងនៅរសៀលថ្ងៃទី ២៩ ខែមករា ឧបនាយករដ្ឋមន្រ្តី លោក Vu Duc Dam បានឲ្យដឹងថា រហូតមកដល់រសៀលថ្ងៃទី ២៩ មករានេះ វៀតណាមរកឃើញករណីឆ្លងកូវីដ ចំនួន ៥៣ នាក់ទៀត។ ថ្វីត្បិតដូច្នេះក៏ដោយ ជាមួយនឹងស្ថានភាពនៃការតាមដានស្នាមនិងអនុវត្តវិធានការបង្ការជាច្រើនដូចបច្ចុប្បន្ន ប្រហែលជាក្នុងរយៈពេល ៨ ថ្ងៃទៀត នឹងទប់ស្កាត់បានការឆ្លងរាលដាលជាពុំខាន។ ឧបនាយករដ្ឋមន្ត្រី លោក Vu Duc Dam បានឲ្យដឹងថា ករណីឆ្លងកូវីដ ១៩ ចំនួន ៥០ នៅរសៀលថ្ងៃទី ២៩ នេះ អាចជាការរកឃើញដ៏ធំបំផុតក្នុងការផ្ទុះឡើងលើកនេះ។ ទាក់ទងនឹងការប្តេជ្ញាចិត្តចំណាយរយៈពេល ១០ ថ្ងៃដើម្បីទប់ស្កាត់ការរីករាលដាល  ឧបនាយករដ្ឋមន្ត្រីលោក Vu Duc Dam បានអះអាងថា រហូតមកដល់ពេលនេះ រដ្ឋាភិបាលនៅតែរក្សាការប្តេជ្ញាចិត្តនេះ។ យោងតាមលោកឧបនាយករដ្ឋមន្រ្តី ការតាមដានស្នាមដែលបានប្រតិបត្តិយ៉ាងបន្ទាន់និងមានប្រសិទ្ធិភាព។")
        print('\nTITLE:\n', article.title)
        print('\nCONTENTS:\n', article.contents)
        print('\nSENTENCES:\n', article.sentences)

    def test_constructor(self):
        article = VietnameseArticle("Trí thức người Việt tại Hàn Quốc mong muốn đóng góp cho quê hương\n Tiến sỹ Vũ Đức Lượng, Phó Chủ tịch Hội người Việt Nam tại Hàn Quốc bày tỏ kỳ vọng Đại hội sẽ đề ra được những định hướng, quyết sách đúng đắn nhằm đưa nước tiếp tục phát triển nhanh và bền vững hơn, tiến tới mục tiêu là nước đang phát triển có công nghiệp hiện đại vào năm 2030. Trong khi đó, anh Vũ Đức Lượng mong muốn trong nhiệm kỳ tới, Đảng và Nhà nước sẽ có thêm nhiều chính sách thể hiện hơn nữa sự quan tâm của lãnh đạo đất nước tới cộng đồng trí thức, chuyên gia người Việt - gốc Việt, kiều bào ở nước ngoài nói chung, phát huy sức mạnh đại đoàn kết toàn dân tộc để kiều bào có thể đóng góp nhiều hơn nữa, tốt hơn nữa, góp sức vào công cuộc xây dựng và phát triển đất nước ngày càng giàu mạnh, phồn vinh, để mọi người dân Việt Nam dù ở trong hay ngoài nước đều tự hào khi mình mang trong người “dòng máu Việt”. Chia sẻ về ý nghĩa và tầm quan trọng của Đại hội Đảng lần này, anh Trần Thiện Quang, Chủ tịch Hội Sinh viên Việt Nam tại Hàn Quốc cho biết anh luôn tin tưởng vào sự lãnh đạo của Đảng, mong muốn được cống hiến hết mình vào sự phát triển của đất nước.")
        print('\nTITLE:\n', article.title)
        print('\nCONTENTS:\n', article.contents)
        print('\nSENTENCES:\n', article.sentences)


if __name__ == '__main__':
    unittest.main()
