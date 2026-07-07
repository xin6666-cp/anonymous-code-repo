# models package
#
# 注意:
#   1. 所有模块都使用 `from models.<sub> import <cls>` 的形式直接从子模块导入,
#      不依赖此 __init__.py 做 re-export, 所以这里保持空即可。
#   2. 之前残留的 `from .visual_sentence_alignment import VisualSentenceAlignment`
#      已删除 -- VSA 模块整体已被 SPR (soft_prior_registration) 取代。
