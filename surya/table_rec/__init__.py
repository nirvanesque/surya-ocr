from collections import defaultdict
from typing import List

from PIL import Image

from surya.common.predictor import BasePredictor
from surya.layout.schema import LayoutResult
from surya.settings import settings
from surya.foundation import FoundationPredictor, TaskNames
from surya.foundation.util import prediction_to_polygon_batch
from surya.input.processing import convert_if_not_rgb
from surya.table_rec.schema import TableRow, TableCell, TableResult, TableCol


class TableRecPredictor(BasePredictor):
    batch_size = settings.TABLE_REC_BATCH_SIZE
    default_batch_sizes = {"cpu": 4, "mps": 4, "cuda": 32, "xla": 16}

    # Override base init - Do not load model
    def __init__(self, foundation_predictor: FoundationPredictor):
        self.foundation_predictor = foundation_predictor
        self.processor = self.foundation_predictor.processor
        self.bbox_size = self.foundation_predictor.model.config.bbox_size
        self.tasks = self.foundation_predictor.tasks

    def __call__(
        self, images: List[Image.Image], batch_size: int | None = None, top_k: int = 5
    ) -> List[LayoutResult]:
        assert all([isinstance(image, Image.Image) for image in images])

        images = convert_if_not_rgb(images)
        images = [self.processor.image_processor(image) for image in images]

        predicted_tokens, batch_bboxes, scores, _ = (
            self.foundation_predictor.prediction_loop(
                images=images,
                input_texts=["" for _ in range(len(images))],
                task_names=[TaskNames.table_structure for _ in range(len(images))],
                batch_size=batch_size,
                max_lookahead_tokens=0,  # Do not do MTP for tables
            )
        )

        image_sizes = [img.shape for img in images]
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, image_sizes, self.bbox_size, self.bbox_size // 2
        )
        table_results = []
        for image, image_tokens, image_polygons, image_scores in zip(
            images, predicted_tokens, predicted_polygons, scores
        ):
            table_rows = []
            table_cells = []
            current_row = None
            current_cell = None
            colspan = 1
            rowspan = 1
            row_id = -1
            cell_id = -1
            within_row_id = -1

            row_has_header = False
            previous_predicted_label = None
            for z, (tok, poly, score) in enumerate(
                zip(image_tokens, image_polygons, image_scores)
            ):
                if tok == self.processor.eos_token_id:
                    break

                predicted_label = self.processor.decode([tok], "table_structure")
                if predicted_label == "</table-row>":
                    if current_row is not None:
                        current_row.is_header = row_has_header
                        table_rows.append(current_row)

                    row_id += 1
                    within_row_id = -1
                    has_header = False
                    current_row = TableRow(
                        row_id=row_id,
                        is_header=has_header,
                        polygon=poly.tolist(),
                    )
                elif predicted_label in ["<table-cell>", "<header-cell>"]:
                    if current_cell is not None:
                        current_cell.colspan = colspan
                        current_cell.rowspan = rowspan
                        table_cells.append(current_cell)

                    colspan = 1
                    rowspan = 1
                    within_row_id += 1
                    cell_id += 1
                    cell_is_header = predicted_label == "<header-cell>"
                    row_has_header = row_has_header or cell_is_header

                    current_cell = TableCell(
                        colspan=colspan,
                        rowspan=rowspan,
                        within_row_id=within_row_id,
                        row_id=row_id,
                        cell_id=cell_id,
                        is_header=cell_is_header,
                        polygon=poly.tolist(),
                    )
                elif predicted_label == "</table-container>":
                    if current_cell is not None:
                        current_cell.colspan = colspan
                        current_cell.rowspan = rowspan
                        table_cells.append(current_cell)
                elif predicted_label in ["<table-cell-columns>", "<table-cell-rows>"]:
                    pass
                else:
                    if (
                        previous_predicted_label == "<table-cell-columns>"
                        and current_cell is not None
                    ):
                        current_cell.colspan = colspan
                    elif (
                        previous_predicted_label == "<table-cell-rows>"
                        and current_cell is not None
                    ):
                        current_cell.rowspan = rowspan

                previous_predicted_label = predicted_label

            cell_columns = defaultdict(list)
            row_id = None
            prev_colspan = 0
            for cell in table_cells:
                if row_id != cell.row_id:
                    row_id = cell.row_id
                    prev_colspan = 0
                cell_columns[prev_colspan].append(cell)
                prev_colspan += cell.colspan

            columns = []
            for col_id in sorted(cell_columns.keys()):
                is_header = any(cell.is_header for cell in cell_columns[col_id])
                col_bbox = [
                    min(cell.bbox[0] for cell in cell_columns[col_id]),
                    min(cell.bbox[1] for cell in cell_columns[col_id]),
                    max(cell.bbox[2] for cell in cell_columns[col_id]),
                    max(cell.bbox[3] for cell in cell_columns[col_id]),
                ]
                col_polygon = [
                    [col_bbox[0], col_bbox[1]],
                    [col_bbox[2], col_bbox[1]],
                    [col_bbox[2], col_bbox[3]],
                    [col_bbox[0], col_bbox[3]],
                ]
                columns.append(
                    TableCol(col_id=col_id, is_header=is_header, polygon=col_polygon)
                )

            result = TableResult(
                cells=table_cells,
                rows=table_rows,
                cols=columns,
                image_bbox=[0, 0, *image.shape],
            )
            table_results.append(result)

        assert len(table_results) == len(images)
        return table_results
