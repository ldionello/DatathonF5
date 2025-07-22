import streamlit as st
import pandas as pd
from modelo import carregar_base, treinar_modelo, prever_candidato

# Sua lista de palavras-chave (copie e cole inteira no código final)
palavras_chave = [
    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin',
    'swift', 'typescript', 'matlab', 'scala', 'perl', 'cobol', 'fortran', 'pascal',
    'visual basic', 'vba', 'objective-c', 'dart', 'julia', 'erlang', 'haskell', 'clojure',
    'f#', 'groovy', 'lua', 'assembly', 'delphi', 'vb.net', 'powershell',
    'sql', 'mysql', 'postgresql', 'oracle', 'sql server', 'mongodb', 'redis', 'cassandra',
    'elasticsearch', 'sqlite', 'mariadb', 'dynamodb', 'neo4j', 'couchdb', 'influxdb',
    'firebase', 'snowflake', 'bigquery', 'redshift', 'teradata', 'db2', 'sybase',
    'banco de dados', 'nosql', 'acid', 'oltp', 'olap', 'data warehouse', 'etl',
    'aws', 'azure', 'google cloud', 'gcp', 'cloud computing', 'ec2', 's3', 'lambda',
    'docker', 'kubernetes', 'terraform', 'ansible', 'chef', 'puppet', 'jenkins',
    'gitlab ci', 'github actions', 'devops', 'ci/cd', 'iac', 'serverless', 'microservices',
    'containerização', 'orquestração', 'saas', 'paas', 'iaas',
    'excel', 'word', 'powerpoint', 'outlook', 'access', 'visio', 'project', 'teams',
    'sharepoint', 'onedrive', 'office 365', 'microsoft 365', 'power apps', 'power automate',
    'power bi', 'tableau', 'qlik', 'looker', 'dax', 'power query', 'ssis', 'ssas', 'ssrs',
    'pentaho', 'cognos', 'microstrategy', 'spotfire', 'alteryx', 'knime', 'talend',
    'data analytics', 'business intelligence', 'bi', 'kpi', 'dashboard', 'relatórios',
    'etl', 'elt', 'data mining', 'olap', 'cubo', 'dimensão', 'fato',
    'big data', 'hadoop', 'spark', 'kafka', 'storm', 'hive', 'pig', 'hbase', 'zookeeper',
    'data science', 'machine learning', 'deep learning', 'ai', 'inteligência artificial',
    'neural networks', 'redes neurais', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
    'numpy', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'spyder',
    'data lake', 'data warehouse', 'data mart', 'data pipeline', 'streaming',
    'linux', 'windows', 'macos', 'unix', 'ubuntu', 'centos', 'red hat', 'debian',
    'fedora', 'suse', 'android', 'ios', 'shell script', 'bash', 'cmd', 'powershell',
    'virtualização', 'vmware', 'hyper-v', 'virtualbox',
    'tcp/ip', 'dns', 'dhcp', 'vpn', 'firewall', 'ips', 'ids', 'ssl', 'tls', 'https',
    'cibersegurança', 'segurança da informação', 'ethical hacking', 'penetration testing',
    'cryptografia', 'blockchain', 'cisco', 'juniper', 'fortinet', 'palo alto',
    'wi-fi', 'lan', 'wan', 'vlan', 'routing', 'switching', 'load balancer',
    'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django',
    'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'ajax', 'rest api',
    'graphql', 'soap', 'json', 'xml', 'mvc', 'orm', 'crud', 'frontend', 'backend',
    'full stack', 'responsive design', 'sass', 'less', 'webpack', 'npm', 'yarn',
    'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'devops', 'itil', 'cobit',
    'pmbok', 'prince2', 'six sigma', 'design thinking', 'ux/ui', 'user experience',
    'user interface', 'wireframe', 'prototype', 'mockup',
    'sap', 'oracle erp', 'microsoft dynamics', 'peoplesoft', 'jd edwards', 'workday',
    'salesforce', 'crm', 'erp', 'scm', 'wms', 'mes', 'bi', 'bpm', 'workflow',
    'enterprise architecture', 'togaf', 'zachman',
    'teste de software', 'qa', 'qc', 'selenium', 'junit', 'testng', 'cucumber',
    'postman', 'soap ui', 'jmeter', 'load testing', 'performance testing',
    'automation testing', 'manual testing', 'bug tracking', 'jira', 'bugzilla',
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'azure devops',
    'version control', 'branching', 'merging', 'pull request', 'code review',
    'iot', 'internet das coisas', 'arduino', 'raspberry pi', 'mqtt', 'edge computing',
    'realidade aumentada', 'realidade virtual', '5g', 'blockchain',
    'smart city', 'industria 4.0', 'digital twin', 'quantum computing',
    'lgpd', 'gdpr', 'sox', 'iso 27001', 'iso 20000', 'pci dss', 'hipaa',
    'governança de ti', 'compliance', 'auditoria', 'gestão de riscos',
    'continuidade de negócios', 'disaster recovery', 'backup',
    'cissp', 'cism', 'cisa', 'pmp', 'itil foundation', 'aws certified',
    'azure certified', 'google cloud certified', 'oracle certified',
    'microsoft certified', 'cisco certified', 'comptia', 'cissp',
    'api', 'webhook', 'microservices', 'soap', 'rest', 'graphql', 'grpc',
    'message queue', 'redis', 'memcached', 'nginx', 'apache', 'iis',
    'load balancer', 'cdn', 'cache', 'session', 'cookies', 'jwt',
    'oauth', 'saml', 'ldap', 'active directory', 'single sign-on', 'sso']


@st.cache_resource
def carregar_modelo():
    try:
        df = pd.read_csv('dados/arquivo.csv', sep=';', engine='python')
    except pd.errors.ParserError:
        df = pd.read_csv('dados/arquivo.csv', sep=',', engine='python')

    return treinar_modelo(df)


modelo = carregar_modelo()

st.title('Preditor de Contratação - Fase 5')

nivel_academico = st.selectbox('Nível Acadêmico', [
                               'Ensino Superior Completo', 'Ensino Médio', 'Pós-Graduação', 'Outro'])
nivel_ingles = st.selectbox(
    'Nível de Inglês', ['Nenhum', 'Básico', 'Intermediário', 'Avançado', 'Fluente'])
nivel_espanhol = st.selectbox('Nível de Espanhol', [
                              'Nenhum', 'Básico', 'Intermediário', 'Avançado', 'Fluente'])
area_atuacao = st.multiselect('Área de Atuação', ['TI - Desenvolvimento/Programação-', 'TI - SAP-',
                              'TI - Projetos-', 'TI - Infraestrutura-', 'Gestão e Alocação de Recursos de TI-', 'Outro'])
tipo_contratacao = st.selectbox(
    'Tipo de Contratação', ['CLT Full', 'PJ/Autônomo', 'Outro'])
nivel_profissional = st.selectbox(
    'Nível Profissional', ['Pleno', 'Sênior', 'Júnior'])

keywords_encontradas = st.multiselect(
    'Skills Canditatos', sorted(palavras_chave))
keywords_vaga = st.multiselect(
    'Skill da Vaga', sorted(palavras_chave))

if st.button('Prever'):
    input_data = pd.DataFrame([{
        'keywords_encontradas': ' '.join(keywords_encontradas.split()) if isinstance(keywords_encontradas, str) else '',
        'keywords_vaga': ' '.join(keywords_vaga),
        'nivel_academico': nivel_academico,
        'nivel_ingles': nivel_ingles,
        'nivel_espanhol': nivel_espanhol,
        'area_atuacao': ', '.join(area_atuacao),
        'tipo_contratacao': tipo_contratacao,
        'nivel_profissional': nivel_profissional
    }])

    pred, proba = prever_candidato(modelo, input_data)
    st.success(f'Probabilidade de contratação: {proba[0]:.2%}')
