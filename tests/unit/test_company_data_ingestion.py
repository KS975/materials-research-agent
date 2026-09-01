from pathlib import Path
import json, zipfile
import pytest
from openpyxl import Workbook
from company_data import CompanyDataRepository, CompanyDataValidationError, import_company_archive
from runtime.company_data_ui import build_company_data_overview

def _xlsx(path, headers, rows):
    wb=Workbook(); ws=wb.active; ws.append(headers)
    for row in rows: ws.append(row)
    path.parent.mkdir(parents=True,exist_ok=True); wb.save(path)

def build_archive(tmp_path):
    root=tmp_path/'src'/'海科数据整理'/'总库'
    _xlsx(root/'样品.xlsx',['样品名称','产品类型','工艺'],[['样品1','P1','合成工艺'],['样品2','P1','合成工艺'],['样品3','P2','合成工艺']])
    _xlsx(root/'配方.xlsx',['样品名称',1001,1002],[['样品1',60,40],['样品2',55,45],['样品3',80,20]])
    _xlsx(root/'测试性能.xlsx',['样品名称','冲击强度','MFR'],[['样品1',12,8],['样品2',14,7],['样品3',20,5]])
    _xlsx(root/'工艺.xlsx',['样品名称','LOGINCATEGORY','TASKCATEGORY'],[['样品1','finishedProduct','measure'],['样品2','finishedProduct','measure'],['样品3','finishedProduct','measure']])
    _xlsx(root/'测试条件.xlsx',['样品名称','测试性能','测试条件','测试条件取值','测试条件类型'],[])
    _xlsx(root/'服役性能.xlsx',['样品名称','寿命'],[])
    _xlsx(root/'服役条件.xlsx',['样品名称','服役性能','服役条件','服役条件取值','服役条件类型'],[])
    _xlsx(root/'原料模板.xlsx',['名称','备注'],[[1001,None],[1002,None]])
    subset=tmp_path/'src'/'海科数据整理'/'P1'; _xlsx(subset/'样品.xlsx',['样品名称','产品类型','工艺'],[['S1','P1','合成工艺']])
    archive=tmp_path/'company.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in (tmp_path/'src').rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(tmp_path/'src').as_posix())
    return archive

def test_import_and_manifest(tmp_path):
    r=import_company_archive(source_zip=build_archive(tmp_path),runtime_root=tmp_path/'runtime'); s=r['manifest']['summary']
    assert (s['samples'],s['products'],s['materials'],s['performance_metrics'])==(3,2,2,2)
    assert s['explicit_test_condition_rows']==0 and s['material_process_parameter_rows']==0
    assert r['manifest']['safety']['official_model_allowed_from_import_alone'] is False

def test_same_archive_is_idempotent(tmp_path):
    a=build_archive(tmp_path); import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime'); replay=import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime'); assert replay['idempotent_replay'] is True

def test_repository_product_and_ui(tmp_path):
    a=build_archive(tmp_path); import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime'); repo=CompanyDataRepository(tmp_path/'runtime'); assert repo.product(product_name='P1')['sample_count']==2; view=build_company_data_overview(tmp_path/'runtime',product_name='P1'); assert view['kind']=='company_real_data'

def test_modeling_export_is_honest(tmp_path):
    a=build_archive(tmp_path); import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime'); repo=CompanyDataRepository(tmp_path/'runtime'); out=repo.export_modeling_dataset(product_name='P1',target_metric='冲击强度'); reality=json.loads(Path(out['reality_json']).read_text(encoding='utf-8')); assert reality['summary']['formula_present']==2; assert reality['summary']['process_present']==0; assert reality['summary']['conditions_present']==0; assert reality['summary']['core_closed_formula_process_target']==0; assert reality['target']['numeric_count']==2

def test_unknown_product_rejected(tmp_path):
    a=build_archive(tmp_path); import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime'); repo=CompanyDataRepository(tmp_path/'runtime');
    with pytest.raises(CompanyDataValidationError): repo.product(product_name='NO_SUCH_PRODUCT')

def test_zip_slip_is_rejected(tmp_path):
    a=tmp_path/'evil.zip';
    with zipfile.ZipFile(a,'w') as z: z.writestr('../evil.txt','x')
    with pytest.raises(CompanyDataValidationError): import_company_archive(source_zip=a,runtime_root=tmp_path/'runtime')
