import lxml.etree
import requests
import csv
import pymysql

"""
try:
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='prioirty', database='introdase')
    cursor = db.cursor()
except Exception as e:
    print(e)
    exit()
"""

fp = './spider_spec_cfp2017.csv'
with open(fp, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Test Sponsor', 'Cpu Name', 'Max MHz', 'Nominal', 'Oderable', 'Cache L1', 'L2', 'L3', 'Cache Other', 'Memory', 'Storage', 'Hardware Other',
                     'OS', 'Compiler', 'Firmware', 'File System', 'System State', 'Base Pointers', 'Peak Pointers', 'Hardware Other', 'Power Management',
                     'Paralled', 'Base Threads', 'Enabled Cores', 'Enabled Chips', 'Threads/Core', 'Base Results', 'Peak Results']) # write header

urls = ['https://www.spec.org/cpu2017/results/cfp2017.html']
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/69.0.3497.81 Chrome/69.0.3497.81 Safari/537.36'}
for url in urls:
    html = requests.get(url, headers=headers)
    selector = lxml.etree.HTML(html.text)
    infos = selector.xpath("//tbody")
    
    for info in infos:
        for n in range(1, 100):
            try:
                test_sponsor = info.xpath('tr[{}]/td[1]/text()'.format(n))[0]

                sysname_html = info.xpath('tr[{}]/td[2]/span/a[1]/@href'.format(n))[0]
                detail_url = 'https://www.spec.org/cpu2017/results/' + sysname_html
                detail_html = requests.get(detail_url, headers=headers)
                selector2 = lxml.etree.HTML(detail_html.text)

                #hardware
                hardware_info = selector2.xpath("//*[@id='Hardware']/tbody")[0]
                cpu_name = hardware_info.xpath('tr[1]/td/text()')[0]
                max_mhz = hardware_info.xpath('tr[2]/td/text()')[0]
                nominal = hardware_info.xpath('tr[3]/td/text()')[0]
                oderable = hardware_info.xpath('tr[5]/td/text()')[0]
                cache_l1 = hardware_info.xpath('tr[6]/td/text()')[0]
                cache_l2 = hardware_info.xpath('tr[7]/td/text()')[0]
                cache_l3 = hardware_info.xpath('tr[8]/td/text()')[0]
                cache_other = hardware_info.xpath('tr[9]/td/text()')[0]
                memory = hardware_info.xpath('tr[10]/td/text()')[0]
                storage = hardware_info.xpath('tr[11]/td/text()')[0]
                hardware_other = hardware_info.xpath('tr[12]/td/text()')[0]

                #software
                software_info = selector2.xpath("//*[@id='Software']/tbody")[0]
                oslist = software_info.xpath('tr[1]/td//text()')
                os = ';'.join(oslist)
                cplist = software_info.xpath('tr[2]/td//text()')
                compiler = ';'.join(cplist)
                firmware = software_info.xpath('tr[4]/td/text()')[0]
                file_sys = software_info.xpath('tr[5]/td/text()')[0]
                sys_state = software_info.xpath('tr[6]/td/text()')[0]
                base_pointers = software_info.xpath('tr[7]/td/text()')[0]
                peak_pointers = software_info.xpath('tr[8]/td/text()')[0]
                software_other = software_info.xpath('tr[9]/td/text()')[0]
                try:
                    power_management = software_info.xpath('tr[10]/td/text()')[0]
                except:
                    power_management = 'None'

                parallel = info.xpath('tr[{}]/td[3]/text()'.format(n))[0]
                base_threads = info.xpath('tr[{}]/td[4]/text()'.format(n))[0]
                enable_cores = info.xpath('tr[{}]/td[5]/text()'.format(n))[0]
                enable_chips = info.xpath('tr[{}]/td[6]/text()'.format(n))[0]
                threads_core = info.xpath('tr[{}]/td[7]/text()'.format(n))[0]
                base_results = info.xpath('tr[{}]/td[8]/text()'.format(n))[0].replace('\xa0','')
                peak_results = info.xpath('tr[{}]/td[9]/text()'.format(n))[0].replace('\xa0','')
                #print (test_sponsor, system_name, parallel, base_threads, enable_cores,enable_chips, threads_core, base_results, peak_results)
                with open(fp, 'a', newline='') as file:
                    writer = csv.writer(file)                    
                    writer.writerow([test_sponsor, cpu_name, max_mhz, nominal, oderable, cache_l1, cache_l2, cache_l3, cache_other, memory, storage, hardware_other,
                                     os, compiler, firmware, file_sys, sys_state, base_pointers, peak_pointers, software_other, power_management,
                                     parallel, base_threads, enable_cores, enable_chips, threads_core, base_results, peak_results])

                """
                try:
                    sql = "INSERT INTO introdase.spec(testsponsor, systemname, baseresult) values('%s', '%s', '%s')" % (test_sponsor, system_name, base_results)
                    cursor.execute(sql)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    print(e)
                    continue
                """
            except:
                #db.rollback()
                continue
                
"""
cursor.close()
db.close()
"""

print("spider success!")