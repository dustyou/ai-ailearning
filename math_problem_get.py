# 引用log_config的import语句一定要放在文件的第一行, 否则影响日志配置
# 爬虫下载数学题目
import json
import logging
import colorama
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = None

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s' +colorama.Style.RESET_ALL, level=logging.INFO)
logging.addLevelName(logging.INFO, colorama.Style.NORMAL + colorama.Fore.GREEN + 'INFO ' + colorama.Style.NORMAL)
logging.addLevelName(logging.WARNING, colorama.Style.BRIGHT + colorama.Fore.YELLOW + 'WARNING ' + colorama.Style.BRIGHT)
logging.addLevelName(logging.ERROR, colorama.Style.BRIGHT + colorama.Fore.RED + 'ERROR ' + colorama.Style.BRIGHT)

# 获取 root logger
logger = logging.getLogger()
log_file = './log/test.log'
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s' + colorama.Style.RESET_ALL)
fh.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logging.error(f'log file: {log_file}')

# 记录一条日志
logger.info('info test')
logger.warning('warning test')
logger.error('error test')


class TestMathProblemGet:

    @staticmethod
    def get_chrome_driver():
        global driver
        if driver is not None:
            return driver
        option = webdriver.ChromeOptions()
        # user_dir = AutotestConfig.get_attr('chrome.user.dir')
        # option.add_argument('--user-data-dir=' + user_dir)  # 设置成用户自己的数据目录
        option.add_argument("--test-type")
        option.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(options=option)
        driver.maximize_window()
        return driver


    @staticmethod
    def get_math_things():
        # 结果
        problems = []
        problem_sets = []
        driver = TestMathProblemGet.get_chrome_driver()
        math_home_url = 'https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions&#39;  
        driver.get(math_home_url)
        # 从 math_home_url 中获取每一年的题目链接, 存入 problem_element
        for problem_element in driver.find_elements(By.XPATH, '//td/a[contains(text(),"AIME")]'):
            year = problem_element.find_element(By.XPATH, '../../td').text
            test_num = problem_element.text
            url = problem_element.get_attribute('href')
            problem_sets.append({'year': year, 'test_num': test_num, 'url':url})
        # 遍历每一年的链接, 获取每个题目的链接, 存入 problems
        for item in problem_sets:
            url = item['url']
            logging.info(f'go to {url}')
            driver.get(url)
            problem_elements = driver.find_elements(By.XPATH, '//li/a[contains(text(),"Problem ")]')
            for problem_element in problem_elements:
                problems.append({'title': problem_element.get_attribute('title'), 'url': url})
                logging.info(f"{problem_element.get_attribute('title')} get url")
        with open('problem_url.json', 'w') as f:
            json.dump(problems, f)
        with open('problem_set.json') as f:
            problems = json.load(f)
        # 遍历每一道题目, 获取问题和答案内容
        for item in problems:
            logging.info(f"title: {item['title']}")
            logging.info(f"url: {problems['url']}")
            item['problem'] = ''
            item['solutions'] = []
            driver.get(item['url'])
            root_element = driver.find_element(By.CLASS_NAME, 'mw-parser-output')
            elements = root_element.find_elements(By.XPATH, './*')
            content = ''
            for element in elements:
                text = element.text
                # 判断标题
                if element.tag_name == 'h2':
                    # 如果标题中含有 Solution, Problem已经结束
                    if 'Solution' in text:
                        if not item['problem']:
                            item['problem'] = content
                        else:
                            item['solutions'].append(content)
                        content = ''
                    content += text + '\n'
                elif element.tag_name == 'p':
                    content += element.get_attribute('innerHTML') + '\n'
        # 导出结果
        with open('data.json', 'w') as f:
            json.dump(problems, f)

if __name__ == '__main__':
    TestMathProblemGet.get_math_things()
