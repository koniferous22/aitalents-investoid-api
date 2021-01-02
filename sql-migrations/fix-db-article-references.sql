ALTER TABLE public.article_references
    RENAME index TO id;

ALTER TABLE public.article_references
    ADD CONSTRAINT pk_article_reference PRIMARY KEY (id);

ALTER TABLE public.article_references
    ADD CONSTRAINT fk_article_company FOREIGN KEY (article_entity)
    REFERENCES public.article_companies (id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE NO ACTION
    NOT VALID;
CREATE INDEX fki_fk_article_company
    ON public.article_references(article_entity);