ALTER TABLE public.article_companies 
RENAME COLUMN index TO id;

ALTER TABLE public.article_companies
    ADD CONSTRAINT pk_article_company PRIMARY KEY (id);
ALTER TABLE public.article_companies
    ADD CONSTRAINT uk_article_entity UNIQUE (article_entity);

ALTER TABLE public.article_companies ADD COLUMN search_index tsvector;
UPDATE public.article_companies SET search_index =
    to_tsvector(
        'english',
        coalesce(article_entity,'') || ' ' ||
        coalesce(nasdaq_entity,'') || ' ' ||
        coalesce(nyse_entity,'') || ' ' ||
        coalesce(nasdaq_label,'') || ' ' ||
        coalesce(nyse_label,'')
    );

CREATE INDEX gin_search_index ON public.article_companies USING GIN (search_index);
