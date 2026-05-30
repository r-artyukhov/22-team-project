## Анализ логирования Hadoop (hdfs)

На текущий момент (май 2026) актуальной версией Hadoop 3.5.

Данные с используемого нами датасета отностся к версии 0.Х (предположительно 0.2), данные собраны в 2008 году. 

За 18 лет архитектура Hadoop перерабатывалась, в том числе поменялось и логирование. На текущий момент логирование построено на log4j фреймворке.

Это означает, что события в логах в версии 0.Х значительно отличаются по своей структуре и генерирующими модулями от актуальной версии, хоть и log4j имеет достаточно гибкую возможность настройки формата логов.

В логах современной версии можно наблюдать все те же, типы событий, что и в логах 0.Х. Так как большинство построенных моделей работают на типах событиях, в том числе последовательностях, то это означает возможность применения нашей модели к современным логам hadoop

Пример сопоставления логов:

| Типы событий, паттерн   | Аналог в версии 3.5 |
| ------------- |:-------------:|
| E5: [\*]Receiving block[\*]src:[\*]dest:[\*], E6: [\*]Received block[\*]src:[\*]dest:[\*]of size[\*]       |2026-05-28 20:31:15 INFO  DataNode:759 - Receiving BP-2070526524-172.26.0.   | 2-1779996250359:blk_1073741827_1003 src: /172.26.0.3:52604 dest: /172.26.0.4:9866
| E11,[\*]PacketResponder[\*]for block[\*]terminating[\*]     | 2026-05-28 20:31:06 INFO  DataNode:1550 - PacketResponder: BP-2070526524-172.26.0.2-1779996250359:blk_1073741827_1003, type=LAST_IN_PIPELINE terminating     |
| E17,[\*]:Failed to transfer[\*]to[\*]got[\*]     | 2026-05-28 20:51:19 WARN  DataNode:3130 - DatanodeRegistration(172.26.0.3:9866, datanodeUuid=02e7e7d4-0d30-4577-bb05-b1aeb7d978ec, infoPort=9864, infoSecurePort=0, ipcPort=9867, storageInfo=lv=-57;cid=CID-f1089948-4b8f-45b9-96e5-7b54c3021741;nsid=733790722;c=1779996250367):Failed to transfer BP-2070526524-172.26.0.2-1779996250359:blk_1073741842_1018 to 172.26.0.4:9866 got     |
| E18,[\*]Starting thread to transfer block[\*]to[\*]    | 2026-05-28 21:03:46 INFO  DataNode:2885 - DatanodeRegistration(172.26.0.3:9866, datanodeUuid=02e7e7d4-0d30-4577-bb05-b1aeb7d978ec, infoPort=9864, infoSecurePort=0, ipcPort=9867, storageInfo=lv=-57;cid=CID-f1089948-4b8f-45b9-96e5-7b54c3021741;nsid=733790722;c=1779996250367) Starting thread to transfer BP-2070526524-172.26.0.2-1779996250359:blk_1073741840_1016 to 172.26.0.4:9866     |
| E21,[\*]Deleting block[\*]file[\*]   |2026-05-28 20:04:47 INFO  FsDatasetAsyncDiskService:367 - Deleted BP-2070526524-172.26.0.2-1779996250359 blk_1073741826_1001 URI file:/tmp/hadoop-root/dfs/data/current/BP-2070526524-172.26.0.2-1779996250359/current/finalized/subdir0/subdir0/blk_1073741826    |
| E23,[\*]BLOCK\* NameSystem[\*]delete:[\*]is added to invalidSet of[\*]      | 2026-05-28 21:02:57 INFO  BlockStateChange:2987 - BLOCK* processReport 0xb9ca9a50993d0025 with lease ID 0x985f1c67cb7b505a: from storage DS-dd96d07b-d297-455a-af67-be38e3605dae node DatanodeRegistration(172.26.0.4:9866, datanodeUuid=f13c4bec-f8aa-4223-b10b-5041204f30dc, infoPort=9864, infoSecurePort=0, ipcPort=9867, storageInfo=lv=-57;cid=CID-f1089948-4b8f-45b9-96e5-7b54c3021741;nsid=733790722;c=1779996250367), blocks: 3, hasStaleStorage: false, processing time: 1 msecs, invalidatedBlocks: 0     |
| E26,[\*]BLOCK\* NameSystem[\*]addStoredBlock: blockMap updated:[\*]is added to[\*]size[\*]     | 2026-05-28 20:31:15 INFO  BlockStateChange:3806 - BLOCK* addStoredBlock: 172.26.0.4:9866 is added to blk_1073741827_1003 