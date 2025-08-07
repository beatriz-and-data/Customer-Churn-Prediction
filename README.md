# Telco Customer Churn Prediction
![Descrição da Imagem](heading.png)

# Introdução

**Descrição**

Esse é um projeto de aprendizado de máquina supervisionado que utiliza classificação para prever a probabilidade de churn dos consumidores de uma companhia de telecomunicações fictícia.

**Metodologia**

O framework padrão adotado em meus projetos é o **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Trata-se de uma metodologia amplamente utilizada para guiar projetos de mineração e ciência de dados, fornecendo uma abordagem estruturada e comprovada para conduzir esses projetos desde o início até a implantação.      

CRISP-DM enfatiza fortemente a compreensão do problema de negócio desde o início, garantindo que os esforços de análise de dados estejam alinhados com os objetivos organizacionais e forneçam insights acionáveis. Além disso, oferece um roteiro claro e estruturado que é capaz de orientar qualquer pessoa ou equipe pelas complexidades de um projeto de mineração de dados, do início ao fim.

**Pipeline**

O processo de mineração dos dados é dividido em 6 etapas:
1.  Compreensão do negócio
2.  Compreensão dos dados
    
3.  Preparação dos dados
    
4.  Modelagem
    
5.  Avaliação
    

**Tecnologias e ferramentas**

O projeto foi desenvolvido em Jupyter Notebook com a linguagem Python, e as bibliotecas Pandas, Scikit-learn, Seaborn e Matplotlib.

# Projeto

## 1. Compreensão do negócio

  

— **Qual é o problema de negócios?**

O terceiro trimestre do presente ano fiscal apresentou um desafio significativo para uma fictícia companhia de telefonia, que chamaremos por ‘Telco’. Telco é um provedor proeminente de serviços de telefone fixo e internet que atua na Califórnia. A companhia fornecia esse serviço para mais de 7.000 consumidores no início do trimestre, até que os dados do final desse período revelaram que mais de 25% de seus clientes haviam deixado seus serviços. Logicamente, esse número se tornou uma fonte de grande preocupação para os responsáveis pela empresa.

  

Essa taxa substancial de evasão significa que os clientes não estão satisfeitos ou não encontram mais valor no serviço e por isso estão procurando alternativas na concorrência. As razões para isso podem ser diversas, e entender essas causas é fundamental para a empresa. Para enfrentar esse desafio, o presente projeto de mineração de dados foi solicitado com dois objetivos principais: **1) Identificar as possíveis causas para a saída dos clientes**, **2) Desenvolver um modelo preditivo que seja capaz de prever que um cliente está propenso a deixar a empresa.**

  

Ao final do projeto, a companhia estará apta não somente para tomar decisões bem direcionadas para resolver o problema, como também para prever proativamente o comportamento de seus clientes e otimizar suas estratégias de retenção, garantindo uma base de clientes mais leal e lucrativa.

  

— **Qual o contexto do problema?**

Ao tratar de abandono de clientes, estamos nos referindo ao conceito de churn. Essa é uma métrica empresarial vital que representa a porcentagem de clientes que param de fazer negócios com uma empresa em um período específico (por exemplo, mensal, trimestral ou anual). A taxa de churn de uma empresa é particularmente significativa em setores onde os clientes podem facilmente trocar de provedor, como telecomunicações ou serviços de streaming, como é o caso. Analisar essa métrica possibilita:

  

-   **Indicar a satisfação do cliente:** Uma alta taxa de churn geralmente sugere que os clientes estão insatisfeitos com seus produtos, serviços ou experiência geral.
    
-   **Identificar problemas:** O monitoramento da taxa de churn ajuda as empresas a identificar problemas com suas ofertas, atendimento ao cliente ou posicionamento competitivo.
    
-   **Embasar estratégias:** Entender os fatores que impulsionam a taxa de churn permite que as empresas desenvolvam estratégias direcionadas para melhorar a retenção, o desenvolvimento de produtos e a experiência do cliente.
    

  

Ao lidar com a rotatividade de clientes, é crucial olhar além da taxa de churn em si e analisar uma variedade de Indicadores Chave de Desempenho (KPIs) relacionados. Esses KPIs podem fornecer uma visão mais holística dos motivos pelos quais os clientes estão saindo e quais melhorias podem ser feitas:

  

-   **Índice de Satisfação do Cliente (CSAT)**: Mede a satisfação direta com um produto ou serviço específico. Para empresas de telecomunicações, isso pode ser a satisfação após uma chamada de suporte, uma instalação ou a experiência com a velocidade da internet. Pontuações baixas de CSAT em problemas recorrentes (por exemplo, internet lenta, chamadas interrompidas, problemas de faturamento) são os principais fatores de risco de rotatividade.
    
-   **Valor de Vida Útil do Cliente (CLV)**: É a receita total que se espera que um cliente gere ao longo do relacionamento com a sua empresa. Churn impacta diretamente o CLV, já que clientes perdidos não contribuem com nenhuma receita futura. Analisar o CLV de clientes perdidos em comparação com os retidos pode destacar o custo financeiro da rotatividade.
    

  
  

— **Quais são os benefícios esperados?**

Em termos de negócios, espera-se que esse projeto seja capaz de:

  

-   **Identificar os principais fatores de risco de churn:** Descobrir quais características ou comportamentos dos clientes estão mais fortemente ligados à saída da empresa, fornecendo insights cruciais para a Telco.
    
-   **Identificar clientes de alto risco:** Entender o perfil dos clientes que têm uma alta probabilidade de cancelar seus serviços em breve. Essa segmentação é fundamental para direcionar os esforços de retenção.
    
- **Fornecer um modelo interpretável:** Em vez de simplesmente gerar uma pontuação de risco, o objetivo é construir um modelo que permita que os stakeholders - como gerentes de produto, equipes de marketing e representantes de atendimento ao cliente - compreendam as razões subjacentes para a previsão.
  

Em uma visão voltada para os dados, a empresa definiu que considerará o projeto bem-sucedido se o modelo preditivo alcançar simultaneamente:

  

1.  Recall de no mínimo 70% para os casos de clientes que deram churn.
    
2.  Precisão de no mínimo 60% para os casos de clientes previstos como churn.
    
3.  AUC-ROC superior a 0.75
    

  
